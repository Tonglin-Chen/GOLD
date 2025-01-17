import math
import os.path
import argparse

import torch
import torchvision.utils as vutils

from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datetime import datetime
import random
import yaml

# from ocl import OCL
from preocl_vqvae import ObjVQVAE

from data_img_h5 import GlobVideoDataset
from utils import cosine_anneal, linear_warmup

from torchvision.utils import save_image, make_grid

# from torchsummary import summary 
import pdb
from metrics import *

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--num_vis',type=int, default=4)
parser.add_argument('--num_tests',type=int, default=1)
# Model
parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='/home/usr/datset/gso.h5')
parser.add_argument('--data_name',)
parser.add_argument('--model_name', )
# Training
parser.add_argument('--lr_enc', type=float, default=4e-5)
parser.add_argument('--lr_dec', type=float, default=4e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=10000)
parser.add_argument('--lr_half_life', type=int, default=100000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--steps', type=int, default=500000)
# DINO
parser.add_argument('--d_dino', type=int, default=768)
parser.add_argument('--num_dino_patches', type=int, default=784)
# Slot Attention
parser.add_argument('--num_iterations', type=int, default=2)
parser.add_argument('--num_slots', type=int, default=8)
parser.add_argument('--cnn_hidden_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=64)
parser.add_argument('--mlp_hidden_size', type=int, default=512)
# Background
parser.add_argument('--bck_size', type=int, default=8)
parser.add_argument('--reg_start', type=float, default=0.1)
parser.add_argument('--reg_final', type=float, default=0.1)
parser.add_argument('--reg_steps', type=int, default=300000)

# Identity
parser.add_argument('--int_size', type=int, default=58)
parser.add_argument('--ext_size', type=int, default=6)
parser.add_argument('--num_cls', type=int, default=10)
parser.add_argument('--coef_pi', type=float, default=0.0002)



# Pretrained OCL
parser.add_argument('--ocl_ckp_path',)

# VQVAE Model
parser.add_argument('--vqvae_config_path', default='vqvae/configs/config.yaml')
parser.add_argument('--lr_vqvae', type=float, default=1e-3)
parser.add_argument('--warmup_steps_pct', type=float, default=0.05)
parser.add_argument('--d_vqvae', type=int, default=3)

parser.add_argument('--gold_ckp_path',)

parser.add_argument('--local_rank', default=-1)

args = parser.parse_args()


# Load VQVAE hype-parameters
with open(args.vqvae_config_path) as f:
        config = yaml.safe_load(f)

for key, val in args.__dict__.items():
    if key not in config or val is not None:
        config[key] = val


if args.seed is None:
    args.seed = random.randint(0, 0xffffffff)
torch.manual_seed(args.seed)


# Loading Dataset
test_dataset = GlobVideoDataset(root=args.data_path, phase='test', data_name=args.data_name)
print(f'Loading {len(test_dataset)} videos for validation...')

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': None,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}
test_loader = DataLoader(test_dataset, sampler=None, **loader_kwargs)
test_epoch_size = len(test_loader)


# Load Model
model = ObjVQVAE(config, args)
if os.path.isfile(args.gold_ckp_path):
    checkpoint = torch.load(args.gold_ckp_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f'Load trained model from {args.gold_ckp_path}')
else:
    FileNotFoundError
model = model.cuda()


# Save path
log_path = os.path.join('./outs', args.data_name, args.model_name)
if os.path.exists(log_path):
    print('{} had been created'.format(log_path))
else:
    os.makedirs(log_path)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)


with open(os.path.join(log_path,'hyperparameters.txt'), 'w') as file:
    json.dump(arg_str, file)


with torch.no_grad():
    model.eval()

    vis_protos_save_path = os.path.join('outs', args.data_name, args.model_name, 'vis_protos')
    if os.path.exists(vis_protos_save_path):
        print('{} had been created'.format(vis_protos_save_path))
    else:
        os.makedirs(vis_protos_save_path)

    vis_objrec_save_path = os.path.join('outs', args.data_name, args.model_name, 'vis_objrec')
    if os.path.exists(vis_objrec_save_path):
        print('{} had been created'.format(vis_objrec_save_path))
    else:
        os.makedirs(vis_objrec_save_path)

    vis_specobj_save_path = os.path.join('outs', args.data_name, args.model_name, 'vis_specobj')
    if os.path.exists(vis_specobj_save_path):
        print('{} had been created'.format(vis_specobj_save_path))
    else:
        os.makedirs(vis_specobj_save_path)

    vis_disent_save_path = os.path.join('outs', args.data_name, args.model_name, 'vis_disent')
    if os.path.exists(vis_disent_save_path):
        print('{} had been created'.format(vis_disent_save_path))
    else:
        os.makedirs(vis_disent_save_path)


    for idx in range(args.num_tests):
        print(f'Begin test-{idx}')

        for batch, (video, seg, clss) in enumerate(test_loader):
            video = video.cuda()
            B, C, H, W = video.size()

            recon_exc, recon_pro = model.visual_dis(video)
            save_image(recon_exc, os.path.join(vis_disent_save_path, 'GOLD_disent_{}.png'.format(batch)), nrow=args.batch_size, padding=2, pad_value=1)
            save_image(recon_pro, os.path.join(vis_protos_save_path, 'GOLD_protos_{}.png'.format(batch)), nrow=args.num_cls, padding=2, pad_value=1)

            objs = model.visual_singleobjrec(video)
            save_image(objs.reshape(-1, C, H, W), os.path.join(vis_objrec_save_path, 'GOLD_singleobj_{}.png'.format(batch)), nrow=args.num_slots+1, padding=2, pad_value=0.5)

            recon_specobj = model.visual_spec_obj(video)
            save_image(recon_specobj_rand, os.path.join(vis_specobj_save_path, 'GOLD_specobj_{}.png'.format(batch)), nrow=args.batch_size, padding=2, pad_value=1)

    print('Testing Finished !!!!')




