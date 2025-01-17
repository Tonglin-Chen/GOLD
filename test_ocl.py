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

import random
import yaml

from ocl import OCL
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
parser.add_argument('--num_tests',type=int, default=3)
# Model
parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='/home/usr/dataset/gso.h5')
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



parser.add_argument('--ocl_ckp_path',)

parser.add_argument('--kld_start', type=float, default=0.)
parser.add_argument('--kld_final', type=float, default=1.0)
parser.add_argument('--kld_start_steps', type=int, default=30000)
parser.add_argument('--kld_final_steps', type=int, default=80000)

parser.add_argument('--local_rank', default=-1)

args = parser.parse_args()



if args.seed is None:
    args.seed = random.randint(0, 0xffffffff)
torch.manual_seed(args.seed)

log_path = os.path.join('./logs', args.data_name, args.model_name)
if os.path.exists(log_path):
    print('{} had been created'.format(log_path))
else:
    os.makedirs(log_path)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)

# Save the parameters of GOLD
with open(os.path.join(log_path,'hyperparameters.txt'), 'w') as file:
    json.dump(arg_str, file)


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
model = OCL(args)
if os.path.isfile(args.ocl_ckp_path):
    checkpoint = torch.load(args.ocl_ckp_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f'Load trained model from {args.ocl_ckp_path}')
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


with torch.no_grad():
    model.eval()

    test_outputs = {key: [] for key in ['ami_all_attn', 'ari_all_attn', 'ami_obj_attn', 'ari_obj_attn', 'iou_attn', 'f1_attn',
                                       'ami_all_mask', 'ari_all_mask', 'ami_obj_mask', 'ari_obj_mask', 'iou_mask', 'f1_mask',
                                       'acc',
                                        ]}

    # Save Path
    vis_decomp_save_path = os.path.join('outs', args.data_name, args.model_name, 'vis_decompose')
    if os.path.exists(vis_decomp_save_path):
        print('{} had been created'.format(vis_decomp_save_path))
    else:
        os.makedirs(vis_decomp_save_path)
    
    vis_overlay_save_path = os.path.join('outs', args.data_name, args.model_name, 'vis_overlay')
    if os.path.exists(vis_overlay_save_path):
        print('{} had been created'.format(vis_overlay_save_path))
    else:
        os.makedirs(vis_overlay_save_path)


    for idx in range(args.num_tests):
        print(f'Begin test-{idx}')

        test_output = {key: [] for key in test_outputs}
        pairs_list = []
        for batch, (video, seg, clss) in enumerate(test_loader):
            video = video.cuda()
            B, C, H, W = video.size()
            if seg[0, 0, 0] != 0:
                segment = (seg + 1) % (seg.max() + 1)
            else:
                segment = seg

            (mse, _,_, attns, masks,pi_c) = model(video)

            attn = torch.nn.functional.interpolate(
                attns.flatten(end_dim=1), 
                size=(args.image_size, args.image_size),
                mode='bilinear'
            )   # [B*K, 1, H, W]
            attns = attn.reshape(*attns.shape[:2], 1, args.image_size, args.image_size )
            attns = torch.softmax(attns/0.001, dim=-4)

            mask = torch.nn.functional.interpolate(
                masks.flatten(end_dim=1), 
                size=(args.image_size, args.image_size),
                mode='bilinear'
            )   # [B*K, 1, H, W]
            masks = mask.reshape(*masks.shape[:2], 1, args.image_size, args.image_size )
            masks = torch.softmax(masks/0.001, dim=-4)
            masks_vis = video[:,None] * masks + (1 - masks)

            # calculate the metrics of attention mask 
            attn = attns.reshape(B, args.num_slots, H, W)
            attn = torch.argmax(attn, dim=1)
            ari_ami = compute_ari_ami(segment, attn)
            iou, f1 = compute_iou_f1(segment, attn)
            for key, val in ari_ami.items():
                test_output[key + '_attn'].append(val)
            test_output['iou_attn'].append(iou)
            test_output['f1_attn'].append(f1)

            # calculate the metrics of reconstruction mask 
            mask = masks.reshape(B, args.num_slots, H, W)
            mask = torch.argmax(mask, dim=1)
            ari_ami = compute_ari_ami(segment, mask)
            iou, f1 = compute_iou_f1(segment, mask)
            for key, val in ari_ami.items():
                test_output[key + '_mask'].append(val)
            test_output['iou_mask'].append(iou)
            test_output['f1_mask'].append(f1)

            # Calculate the accuracy of identify
            segs = segment.reshape(B, H, W).cuda()
            mas = masks.reshape(B, args.num_slots, 1, H, W)
            pairs_batch = compute_pairs(video, segs, mas[:, 1:], clss, pi_c)
            pairs_list.append(pairs_batch)

            # Save decomposition and segmentation results
            if idx == 0: 
                visualize_decompose(video, masks, masks_vis, batch, vis_decomp_save_path)
                visualize_seg_overlay_image(video, masks, batch, vis_overlay_save_path)



        id_acc = cluster_acc(args.num_cls, pairs_list)
        test_output['acc'] = id_acc.cpu().numpy()

        test_output = {key: np.array(val).mean() for key, val in test_output.items()}

        logging.info(f'test-{idx}: ' + ', '.join(f'{key}: {val:.6f}' for key, val in test_output.items()))

        # collect the results for each times
        for key, val in test_output.items():
            test_outputs[key].append(val)



    test_outputs = {key: '{:.6f}\u00b1{:.0e}'.format(np.array(val).mean(), np.array(val).std()) for key, val in test_outputs.items()}

    # Save the testing results
    with open(os.path.join(log_path,'metrics_results.txt'), 'w') as file:
        json.dump(test_outputs, file)

    print('====> OCL Test \t\t', ', '.join(f'{key}: {val}' for key, val in test_outputs.items()))


