from utils import *

import pdb
from transformers import ViTFeatureExtractor, ViTModel
import numpy as np
from ocl import OCL
import os
from vqvae import VQVAE1
from torchvision.utils import save_image


def reparameterize_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    return mu + std * noise


class ObjVQVAE(nn.Module):
    
    def __init__(self,config, args):
        super().__init__()
        
        self.num_slots = args.num_slots
        self.slot_size = args.slot_size
        self.img_channels = args.img_channels
        self.image_size = args.image_size
        self.num_dino_patches = args.num_dino_patches
        self.d_vqvae = args.d_vqvae
        self.data_name = args.data_name
        self.int_size = args.int_size
        self.ext_size = args.ext_size
        self.num_cls = args.num_cls
        self.d_dino = args.d_dino

        # Load Pre-trained OCL model
        self.ocl = OCL(args)
        if os.path.exists(args.ocl_ckp_path):
            ckp = torch.load(args.ocl_ckp_path, map_location='cpu')
            if 'state_dict' in ckp:
                ckp = ckp['state_dict']
            # remove 'loss.' keys
            ckp = {k: v for k, v in ckp.items() if not k.startswith('loss.')}
            self.ocl.load_state_dict(ckp)
            print(f'Loaded OCL weight from {args.ocl_ckp_path}!!!')
        else:
            print(f'Warning: OCL weight not found at {args.ocl_ckp_path}!!!')
        # freeze the OCL model
        for p in self.ocl.parameters():
            p.requires_grad = False


        self.vqvae = VQVAE1(config['enc_dec_dict'], config['vq_dict'], use_loss=True)
        self.ds2vq = nn.Sequential(
            linear(args.d_dino, args.mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.mlp_hidden_size, args.d_vqvae)
            )


    def forward(self, video, tau):
        B, C, H, W = video.size()
        
        # Pre-trained OCL 
        with torch.no_grad():
            (slots_int, slots_ext, slots_bck, pi, attns) = self.ocl.encode(video, tau)                   # [B, K+1, C, H, W]
            (patches_all, masks) = self.ocl.decode(slots_int, slots_ext, slots_bck)

            # if self.data_name == 'unique_gso' or  self.data_name == 'octa':
            attns_vis = torch.nn.functional.interpolate(
            attns.flatten(end_dim=1), 
            size=(self.image_size, self.image_size),
            mode='bilinear'
            )   # [B*K, 1, H, W]
            attns_vis = attns_vis.reshape(*attns.shape[:2], 1, self.image_size, self.image_size )
            attns_vis = torch.softmax(attns_vis/0.01, dim=-4)
            #     # obj_flat = (video[:,None] * attns_vis).flatten(end_dim=1).clamp(0., 1.)            # [B*(K+1), C, H, W]
            # else:
            masks_vis = torch.nn.functional.interpolate(
            masks.flatten(end_dim=1), 
            size=(self.image_size, self.image_size),
            mode='bilinear'
            )   # [B*K, 1, H, W]
            masks_vis = masks_vis.reshape(*masks.shape[:2], 1, self.image_size, self.image_size )
            masks_vis = torch.softmax(masks_vis/0.01, dim=-4)
            obj_flat = (video[:,None] * masks_vis).flatten(end_dim=1).clamp(0., 1.)            # [B*(K+1), C, H, W]

        
        # Transform the patche to vqvae emb
        patches_all = self.ds2vq(patches_all)                                      # B, K+1,  num_patches, d_vqvae
        patches = patches_all.permute(0,1,3,2).reshape(B*self.num_slots, self.d_vqvae, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))
        # Decode
        quant, quant_loss, _ = self.vqvae.quantize(patches)
        quant = self.vqvae.post_quant_conv(quant) 
        
        rec_all = self.vqvae.decoder(quant).reshape(B, self.num_slots, self.img_channels+1, self.image_size, self.image_size) 
        rec_apc, rec_mask = rec_all.split([self.img_channels, 1], dim=-3)
        mask_all = torch.softmax(rec_mask, dim=-4)
        recon_objs = rec_apc * mask_all
        recon = (rec_apc * mask_all).sum(-4)  
        
        # Loss
        # VQVAE loss
        loss_dict = self.vqvae.loss(quant_loss, obj_flat, recon_objs.flatten(end_dim=1))
        vqvae_loss = loss_dict['quant_loss'] + loss_dict['percept_loss']
        # Recon loss
        rec_loss = ((video - recon)**2).mean()
        loss = vqvae_loss + rec_loss

        obj_vis = obj_flat.reshape(B, self.num_slots, *obj_flat.shape[1:])

        return (loss, obj_vis, recon_objs, recon, masks_vis, pi, attns_vis)


    def visual_dis(self, video, tau):

        (slots_int, slots_ext, slots_bck,_,_) = self.ocl.encode(video, tau)

        pat_bck, alpha_bck = self.ocl.dec_bck(slots_bck)
        
        int1 = slots_int[:, :1]
        ext1 = slots_ext[:, :1]
        int2 = slots_int[:, 1:2]
        ext2 = slots_ext[:, 1:2]

        slots_ext12 = torch.cat([ext2,ext1,slots_ext[:,2:]],dim=1)
     

        pat, pat_alpha = self.ocl.objdec(slots_int, slots_ext)        # B, K, num_patches, d_dino
        pat12, pat12_alpha = self.ocl.objdec(slots_int, slots_ext12)

        alpha_all   = torch.softmax(torch.cat([alpha_bck, pat_alpha], dim=1), dim=1) 
        alpha12_all = torch.softmax(torch.cat([alpha_bck, pat12_alpha], dim=1), dim=1) 
        pat_all   = (torch.cat([pat_bck, pat], dim=1) * alpha_all)
        pat12_all = (torch.cat([pat_bck, pat12], dim=1) * alpha12_all)

        pat_exc = torch.cat([pat_all, pat12_all],dim=0).flatten(end_dim=1)   #[B*2*K, num_patches, d_dino]


        # Transform the patche to vqvae emb
        pat_exc = self.ds2vq(pat_exc)        #[B*4, num_patches, d_vqvae]                      
        pat_exc = pat_exc.permute(0,2,1).reshape(-1, self.d_vqvae, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))
        # Decode
        quant_exc, _, _ = self.vqvae.quantize(pat_exc)
        quant_exc = self.vqvae.post_quant_conv(quant_exc) 
        recon_exc_all = self.vqvae.decoder(quant_exc).reshape(-1, self.num_slots, self.img_channels+1, self.image_size, self.image_size)  

        rec_exc_apc, rec_exc_mask = recon_exc_all.split([self.img_channels, 1], dim=-3)
        mask_exc_all = torch.softmax(rec_exc_mask, dim=-4)
        recon_exc = (rec_exc_apc * mask_exc_all).sum(-4)  # [B*2, C, H, W]
        
        prior_ext_mu = torch.zeros(slots_ext.shape[0], self.ext_size).cuda()
        prior_ext_logvar = torch.zeros(slots_ext.shape[0], self.ext_size).cuda()
        randn_ext = reparameterize_normal(prior_ext_mu, prior_ext_logvar)

        extr_ext = slots_ext[:,0]
        gen_ext = torch.cat([randn_ext, extr_ext])

        rad_ext = gen_ext[:,None].expand(-1, self.num_cls, -1).reshape(-1, self.ext_size)
        rad_protos = self.ocl.ocl_encoder.protos[None].expand(gen_ext.shape[0], -1, -1).reshape(-1, self.int_size)
        pat_pro, alpha_pro = self.ocl.objdec(rad_protos[:,None], rad_ext[:,None])   #[4*10, 1, num_patches, d_dino]
        pat_pro = (pat_pro * torch.sigmoid(alpha_pro)).squeeze(1)  #[4*10, num_patches, d_dino]

        # Transform the patche to vqvae emb
        pat_pro = self.ds2vq(pat_pro)   #[4*10, num_patches, d_vqvae]
        pat_pro = pat_pro.permute(0,2,1).reshape(-1, self.d_vqvae, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))
        # Decode
        quant_pro, _, _ = self.vqvae.quantize(pat_pro)
        quant_pro = self.vqvae.post_quant_conv(quant_pro) 
        recon_pro_all = self.vqvae.decoder(quant_pro).reshape(-1, self.img_channels+1, self.image_size, self.image_size) 

        rec_pro_apc, rec_pro_mask = recon_pro_all.split([self.img_channels, 1], dim=-3)
        mask_pro_all = torch.sigmoid(rec_pro_mask)
        recon_pro = rec_pro_apc * mask_pro_all


        return recon_exc, recon_pro



    def visual_singleobjrec(self, video, tau):
        B, C, H, W = video.size()
        # Pre-trained OCL 
        with torch.no_grad():
            (slots_int, slots_ext, slots_bck, pi, attns) = self.ocl.encode(video, tau)                   # [B, K+1, C, H, W]
            (patches_all, masks) = self.ocl.decode(slots_int, slots_ext, slots_bck)

            masks_vis = torch.nn.functional.interpolate(
            masks.flatten(end_dim=1), 
            size=(self.image_size, self.image_size),
            mode='bilinear'
            )   # [B*K, 1, H, W]
            masks_vis = masks_vis.reshape(*masks.shape[:2], 1, self.image_size, self.image_size )
            masks_vis = torch.softmax(masks_vis/0.01, dim=-4)
            obj_flat = (video[:,None] * masks_vis).flatten(end_dim=1).clamp(0., 1.)            # [B*(K+1), C, H, W]

        
        # Transform the patche to vqvae emb
        patches_all = self.ds2vq(patches_all)                                      # B, K+1,  num_patches, d_vqvae
        patches = patches_all.permute(0,1,3,2).reshape(B*self.num_slots, self.d_vqvae, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))
        # Decode
        quant, quant_loss, _ = self.vqvae.quantize(patches)
        quant = self.vqvae.post_quant_conv(quant) 
        recon = self.vqvae.decoder(quant).reshape(B, self.num_slots, self.img_channels, self.image_size, self.image_size)

        objs = recon * masks_vis + (1 - masks_vis)

        rec_objs = torch.cat([video[:,None], objs],dim=1)

        return rec_objs



    def visual_spec_obj(self, video, tau):
        B, C, H, W = video.size()
        (slots_int, slots_ext, slots_bck,_,_) = self.ocl.encode(video, tau)

        # Sample prototypes
        index = torch.randn(self.num_cls-3).cuda()
        ind = torch.argmax(index)
        pro_one = self.ocl.ocl_encoder.protos[ind]  # D
        pro_two = self.ocl.ocl_encoder.protos[ind+1]  # D
        pro_three = self.ocl.ocl_encoder.protos[ind+2]  # D

        # Generante scene by extracting representation
        recon_one_extr = self.gen_sce(slots_ext, slots_bck, 1, pro_one)
        recon_two_extr = self.gen_sce(slots_ext, slots_bck, 2, pro_one, pro_two)
        recon_three_extr = self.gen_sce(slots_ext, slots_bck, 3, pro_one, pro_two, pro_three)

        # Combine three scenes
        recon_specobj = torch.cat([recon_one_extr, recon_two_extr, recon_three_extr])
      
        return recon_specobj

    def gen_sce(self, slots_ext, slots_bck, num_objs, pro_one=None, pro_two=None, pro_three=None):
        B = slots_ext.shape[0]
        if num_objs == 1:
            pros = pro_one[None, None].expand(B, -1, -1)                       # [B, 1, D]
        elif num_objs == 2:
            pros = torch.cat([pro_one[None], pro_two[None]])                   # [2, D]
            pros = pros[None].expand(B, -1, -1)                                # [B, 2, D]
        elif num_objs == 3:
            pros = torch.cat([pro_one[None], pro_two[None], pro_three[None]])  # [3, D]
            pros = pros[None].expand(B, -1, -1)                                # [B, 2, D]

        ext = slots_ext[:,:num_objs]

        # Decode the patch feature of prototyeps
        pat, pat_alpha = self.ocl.objdec(pros, ext)
        # Decode the patch feature of background
        pat_bck, alpha_bck = self.ocl.dec_bck(slots_bck)

        # Decoder the image of scene
        alpha_all   = torch.softmax(torch.cat([alpha_bck, pat_alpha], dim=1), dim=1) 
        pat_all   = (torch.cat([pat_bck, pat], dim=1) * alpha_all).flatten(end_dim=1)   #[B*(num_objs+1), num_patches, d_dino]

        # Transform the patche to vqvae emb
        pat_exc = self.ds2vq(pat_all)        #[B*(num_objs+1), num_patches, d_vqvae]                      
        pat_exc = pat_exc.permute(0,2,1).reshape(-1, self.d_vqvae, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))
        # Decode
        quant_exc, _, _ = self.vqvae.quantize(pat_exc)
        quant_exc = self.vqvae.post_quant_conv(quant_exc) 
        recon_exc_all = self.vqvae.decoder(quant_exc).reshape(-1, num_objs+1, self.img_channels+1, self.image_size, self.image_size)  

        rec_exc_apc, rec_exc_mask = recon_exc_all.split([self.img_channels, 1], dim=-3)
        mask_exc_all = torch.softmax(rec_exc_mask, dim=-4)
        recon_exc = (rec_exc_apc * mask_exc_all).sum(-4)  # [B, C, H, W]

        return recon_exc








        # 


