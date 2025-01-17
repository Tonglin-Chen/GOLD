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

from preocl_vqvae import ObjVQVAE   # procl_vqvae1: single disentangle, procl_vqvae: scene disentangle
from data_img_h5 import GlobVideoDataset
from utils import cosine_anneal, linear_warmup, CosineAnnealingWarmupRestarts
from torchvision.utils import save_image, make_grid

from metrics import *

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--num_vis',type=int, default=4)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--steps', type=int, default=500000)
# Model
parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='/home/usr/dataset/gso.h5')
parser.add_argument('--data_name',)
parser.add_argument('--model_name', )
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
# Identity
parser.add_argument('--int_size', type=int, default=58)
parser.add_argument('--ext_size', type=int, default=6)
parser.add_argument('--num_cls', type=int, default=10)
parser.add_argument('--coef_pi', type=float, default=0.0002)

parser.add_argument('--tau', type=float, default=0.5)

# Pretrained OCL
parser.add_argument('--ocl_ckp_path',)
# VQVAE Model
parser.add_argument('--vqvae_config_path', default='vqvae/configs/config.yaml')
parser.add_argument('--lr_vqvae', type=float, default=1e-3)
parser.add_argument('--warmup_steps_pct', type=float, default=0.05)
parser.add_argument('--d_vqvae', type=int, default=3)


parser.add_argument('--local_rank', default=-1)

args = parser.parse_args()


# Load VQVAE hype-parameters
with open(args.vqvae_config_path) as f:
        config = yaml.safe_load(f)

for key, val in args.__dict__.items():
    if key not in config or val is not None:
        config[key] = val


dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if args.seed is None:
    args.seed = random.randint(0, 0xffffffff)
torch.manual_seed(args.seed)


# Save Path
log_path = os.path.join('./logs', args.data_name, args.model_name)
if os.path.exists(log_path):
    print('{} had been created'.format(log_path))
else:
    os.makedirs(log_path)


if local_rank == 0:
    arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
    arg_str = '__'.join(arg_str_list)
    writer = SummaryWriter(log_path)
    writer.add_text('hparams', arg_str)

# Load dataset
train_dataset = GlobVideoDataset(root=args.data_path, phase='train', data_name=args.data_name)
val_dataset = GlobVideoDataset(root=args.data_path, phase='valid', data_name=args.data_name)
if local_rank == 0:
    print(f'Loading {len(train_dataset)} videos for training...')
    print(f'Loading {len(val_dataset)} videos for validation...')
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': None,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}
train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)
log_interval = train_epoch_size // 5

#Load Model
model = ObjVQVAE(config, args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
    if local_rank == 0:
        print(f'Load trained model from {args.checkpoint_path}')
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0
    if local_rank == 0:
        print('Starting training ...')

model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = Adam(model.parameters(), lr=args.lr_vqvae)
# optimizer = Adam([
#     {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
# ])

if os.path.isfile(args.checkpoint_path):
    optimizer.load_state_dict(checkpoint['optimizer'])


for epoch in range(start_epoch, args.epochs):
    model.train()

    train_sampler.set_epoch(epoch)
    for batch, (video, seg,clss) in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch


        total_steps = args.epochs * train_epoch_size
        warmup_steps = args.warmup_steps_pct * total_steps
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=args.lr_vqvae,
            min_lr=args.lr_vqvae / 100.,
            warmup_steps=warmup_steps,
        )


        video = video.cuda()
        optimizer.zero_grad()
        (mse, obj_vis, rec_vis, img_vis, _,_,_) = model(video, args.tau)
        loss = mse

        loss.backward()
        optimizer.step()
        scheduler.step()


        with torch.no_grad():
            if batch % log_interval == 0 and local_rank == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} '.format(
                      epoch+1, batch, train_epoch_size, loss.item()))
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)


    with torch.no_grad():
        if local_rank == 0:
            coms = torch.cat((video[:4,None], obj_vis[:4], img_vis[:4,None], rec_vis[:4]), dim=1)
            frame = vutils.make_grid(coms.reshape(-1, *coms.shape[-3:]), nrow=(args.num_slots + 1), pad_value=0.8)
            writer.add_image('TRAIN_segment/epoch={:03}'.format(epoch+1), frame)

            recon_exc, recon_pro = model.module.visual_dis(video, args.tau)
            visdis = make_grid(recon_exc, nrow=args.batch_size, padding=2, pad_value=1)
            writer.add_image('VAL_disent/epoch={:03}'.format(epoch+1), visdis)

            vispro = make_grid(recon_pro, nrow=args.num_cls, padding=2, pad_value=1)
            writer.add_image('VAL_protos/epoch={:03}'.format(epoch+1), vispro)


    with torch.no_grad():
        model.eval()

        test_output = {key: [] for key in [
                                           'ami_all_mask', 'ari_all_mask', 'ami_obj_mask', 'ari_obj_mask', 'iou_mask', 'f1_mask',
                                            ]}
        pairs_list = []
        val_mse = 0.
        for batch, (video, seg, clss) in enumerate(val_loader):
            video = video.cuda()
            if seg[0, 0, 0] != 0:
                segment = (seg + 1) % (seg.max() + 1)
            else:
                segment = seg
            (mse, obj_vis, rec_vis, img_vis, masks,pi,_) = model(video, args.tau)
            val_mse += mse

            # calculate the metrics of reconstruction mask 
            mask = masks.reshape(args.batch_size, args.num_slots, args.image_size, args.image_size)
            mask = torch.argmax(mask, dim=1)
            ari_ami = compute_ari_ami(segment, mask)
            iou, f1 = compute_iou_f1(segment, mask)
            for key, val in ari_ami.items():
                test_output[key + '_mask'].append(val)
            test_output['iou_mask'].append(iou)
            test_output['f1_mask'].append(f1)


            # Calculate the accuracy of identify
            pi_c = torch.argmax(pi, dim=-1)
            segs = segment.reshape(args.batch_size, args.image_size, args.image_size).cuda()
            mas = masks.reshape(args.batch_size, args.num_slots, 1, args.image_size, args.image_size)
            pairs_batch = compute_pairs(video, segs, mas[:, 1:], clss, pi_c)
            pairs_list.append(pairs_batch)

        id_acc = cluster_acc(args.num_cls, pairs_list)
        test_output = {key: np.array(val).mean(-1) for key, val in test_output.items()}
        val_mse /= (val_epoch_size)
        val_loss = val_mse

        

        if local_rank == 0:
            writer.add_scalar('VAL/loss', val_loss, epoch+1)
            print('====> Dataset:{}, Model:{},  Epoch: {:3} \t Loss = {:F}'.format(args.data_name, args.model_name, epoch+1, val_loss))
            print('ACC: {:.4f}'.format(id_acc))
            print('Metrics: ' + ', '.join(f'{key}: {val:.4f}' for key, val in test_output.items()))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            if local_rank == 0:
                torch.save(model.module.state_dict(), os.path.join(log_path, 'best_model.pt'))
            if global_step < args.steps and local_rank == 0:
                torch.save(model.module.state_dict(), os.path.join(log_path, f'best_model_until_{args.steps}_steps.pt'))

        if local_rank == 0:
            writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }

        if local_rank == 0:
            torch.save(checkpoint, os.path.join(log_path, 'checkpoint.pt.tar'))

            print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

if local_rank == 0:
    writer.close()
