from utils import *

import pdb
from transformers import ViTFeatureExtractor, ViTModel
import numpy as np
from vqvae.VQVAE import VQVAEWrapper
from torch.distributions import Beta
from torch.distributions.kl import kl_divergence
from torch.distributions.categorical import Categorical
from torchvision.utils import save_image
from torch.distributions.normal import Normal



def reparameterize_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    return mu + std * noise


def compute_kld_normal(mu, logvar, prior_mu, prior_logvar):
    prior_invvar = torch.exp(-prior_logvar)
    kld = 0.5 * (prior_logvar - logvar + prior_invvar * ((mu - prior_mu).square() + logvar.exp()) - 1)
    return kld.sum(-1)


class DINOEncoder(nn.Module):
    """A wrapper for DINO."""

    def __init__(self,
                 resolution: int,
                 patch_size: int = 8,
                 small_size: bool = True):
        super().__init__()

        self.resolution = resolution
        self.patch_size = patch_size
        self.small_size = small_size
        if self.small_size:
            self.version = 's'
        else:
            self.version = 'b'

        self.dino = ViTModel.from_pretrained(
            f'/home/ctl/conference/iclr2024/dino-vit{self.version}{patch_size}')
        for p in self.dino.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Reduce the CLS token
        out = self.dino(x).last_hidden_state[:, 1:, :]
        # out has shape: [B, H*W, C]

        out = out.reshape(out.shape[0], self.resolution, self.resolution,
                          out.shape[-1])
        # out has shape: [B, H, W, C]

        out = out.permute(0, 3, 1, 2)
        # out has shape: [B, C, H, W]
        return out

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # freeze the DINO model
        self.dino.eval()
        return self




class SlotAttentionVideo(nn.Module):
    
    def __init__(self, num_iterations, num_slots, ext_size, int_size, num_cls,
                 input_size, slot_size, mlp_hidden_size,
                 epsilon=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.full_slots_size = slot_size 
        self.upd_slots_size = slot_size
        self.num_iterations = num_iterations
        self.int_size = int_size
        self.ext_size = ext_size
        self.num_cls = num_cls
        self.slot_size = slot_size
        self.epsilon = epsilon


        # parameters for Gaussian initialization (shared by all slots).
        self.ext_mu = nn.Parameter(torch.Tensor(1, 1, ext_size))
        self.ext_log_sigma = nn.Parameter(torch.Tensor(1, 1, ext_size))
        nn.init.xavier_uniform_(self.ext_mu)
        nn.init.xavier_uniform_(self.ext_log_sigma)

        # parameters for Gaussian initialization (shared by all slots).
        self.int_mu = nn.Parameter(torch.Tensor(1, 1, num_cls))
        self.int_log_sigma = nn.Parameter(torch.Tensor(1, 1, num_cls))
        nn.init.xavier_uniform_(self.int_mu)
        nn.init.xavier_uniform_(self.int_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(self.full_slots_size)
        self.norm_mlp_int = nn.LayerNorm(int_size)
        self.norm_mlp_ext = nn.LayerNorm(ext_size)
        self.norm_protos = nn.LayerNorm(int_size)

        # linear maps for the attention module.
        self.project_q = linear(self.full_slots_size, self.full_slots_size, bias=False)
        self.project_k = linear(input_size, self.full_slots_size, bias=False)
        self.project_v = linear(input_size, self.full_slots_size, bias=False)
        self.project_p = linear(int_size, int_size, bias=False)

        # slot update functions.
        self.gru_int = gru_cell(int_size, int_size)
        self.mlp_int = nn.Sequential(
            linear(int_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, int_size))

        # slot update functions.
        self.gru_ext = gru_cell(ext_size, ext_size)
        self.mlp_ext = nn.Sequential(
            linear(ext_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, ext_size))
        

    def forward(self, slots_bck, inputs, protos, tau):
        B, num_inputs, input_size = inputs.size()

        # # initialize slots
        slots_ext = inputs.new_empty(B, self.num_slots, self.ext_size).normal_()
        slots_ext = self.ext_mu + torch.exp(self.ext_log_sigma) * slots_ext

        slots_int = inputs.new_empty(B, self.num_slots, self.num_cls).normal_()
        logits_pi = self.int_mu + torch.exp(self.int_log_sigma) * slots_int

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k

        protos = self.norm_protos(protos)
        protos = self.project_p(protos)


        for i in range(self.num_iterations):  
         
            
            ext_upd = slots_ext

            # if gumbel:
            pi = F.gumbel_softmax(logits_pi, tau=tau, hard=False)
            # else:
            #     pi = F.softmax(logits_pi, dim=-1)
            slots_int = (pi[:,:,:,None] * protos[None,None]).sum(-2)
            int_upd = slots_int
            slots_obj = torch.cat([slots_int, slots_ext],dim=-1)          
            slots = torch.cat([slots_bck, slots_obj[:,1:]],dim=1)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            attn_logits = torch.matmul(k, q.transpose(-1, -2))
            attn_vis = F.softmax(attn_logits, dim=-1)
            # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn_vis + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            updates_int, updates_ext = updates.split([self.int_size, self.ext_size], dim=-1)

            # ext update
            ext_upd = self.gru_ext(updates_ext.reshape(-1, self.ext_size),
                                ext_upd.reshape(-1, self.ext_size)
                )
            slots_ext = ext_upd.reshape(B, self.num_slots, self.ext_size)
             # use MLP only when more than one iterations
            if i < self.num_iterations - 1:
                slots_ext = slots_ext + self.mlp_ext(self.norm_mlp_ext(slots_ext))
            

            # logits_pi update.
            int_upd = self.gru_int(updates_int.reshape(-1, self.int_size),
                                int_upd.reshape(-1, self.int_size)
                )
            int_upd = int_upd.reshape(B, self.num_slots, self.int_size)
            # use MLP only when more than one iterations
            if i < self.num_iterations - 1:
                int_upd = int_upd + self.mlp_int(self.norm_mlp_int(int_upd))

            logits_pi = torch.matmul(int_upd, protos.transpose(1,0))*self.int_size **(-0.5)


        return (slots_ext[:,1:], logits_pi[:,1:], attn_vis)

class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)



class STEVEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
         
        if args.data_name == 'clevr' or args.data_name == 'shop' or args.data_name == 'octa':
            self.cnn = nn.Sequential(
                Conv2dBlock(args.img_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
                Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                conv2d(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                conv2d(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),       # 64 -> 32
                conv2d(args.cnn_hidden_size, args.d_dino, 5, 1),          # 32 -> 28
            )
            self.pos = CartesianPositionalEmbedding(args.d_dino, int(args.num_dino_patches**0.5))  
        
        self.layer_norm = nn.LayerNorm(args.d_dino)

        self.mlp = nn.Sequential(
            linear(args.d_dino, args.d_dino, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.d_dino, args.d_dino)
            )

        self.savi = SlotAttentionVideo(
            args.num_iterations, args.num_slots, args.ext_size, args.int_size, args.num_cls,
            args.d_dino, args.slot_size, args.mlp_hidden_size,
            )

        self.enc_bck = nn.Sequential(
            linear(args.d_dino*args.num_dino_patches, args.mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.mlp_hidden_size, args.bck_size)
            )

        self.bck_proj = linear(args.bck_size, args.slot_size, bias=False)

        self.protos = nn.Parameter(torch.randn(args.num_cls, args.int_size))

        self.bck2lat = nn.Sequential(
            linear(args.bck_size, args.mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.mlp_hidden_size, 2*args.bck_size)
            )

        self.enc_ext = nn.Sequential(
            linear(args.ext_size, args.mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.mlp_hidden_size, 2*args.ext_size)
            )

        # self.enc_bck_lat = linear(args.bck_size, 2*args.bck_size)
        # self.enc_ext_lat = linear(args.ext_size, 2*args.ext_size)
        # self.enc_logits_pi = linear(args.num_cls, args.num_cls)


class PatchDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
    
        
        self.slot_proj = nn.Linear(args.slot_size, args.d_dino, bias=True)
        nn.init.xavier_uniform_(self.slot_proj.weight)
        nn.init.zeros_(self.slot_proj.bias)

        self.pos_embed = nn.Parameter(
                torch.randn(1, args.num_dino_patches, args.d_dino) * 0.02
            )

        self.decoder_obj = build_mlp(args.d_dino, args.d_dino+1)

        self.decoder_bck = nn.Sequential(
            linear(args.bck_size, args.mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.mlp_hidden_size, args.num_dino_patches * (args.d_dino+1))
            )

        # self.patch_proj = nn.Linear(args.d_dino, args.d_vqvae, bias=False)
        # self.bck_patch_proj = nn.Linear(args.d_dino, args.d_vqvae, bias=False)



class OCL(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.cnn_hidden_size = args.cnn_hidden_size
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.img_channels = args.img_channels
        self.image_size = args.image_size
        self.num_dino_patches = args.num_dino_patches
        self.d_dino = args.d_dino
        self.ext_size = args.ext_size
        self.int_size = args.int_size
        self.num_cls = args.num_cls

        # encoder networks
        self.ocl_encoder = STEVEEncoder(args)
        # decoder networks
        self.ocl_decoder = PatchDecoder(args)

        self.dino_encoder = DINOEncoder(resolution=28, patch_size=8, small_size=False)
        self.dino_feature_extractor = ViTFeatureExtractor.from_pretrained('/home/ctl/conference/iclr2024/dino-vitb8')

        self.beta = Beta(torch.tensor(2.0).cuda(), torch.tensor(2.0).cuda())

        self.register_buffer('prior_logits_pi', torch.full((args.num_cls,), 1/args.num_cls))
        self.register_buffer('prior_ext_mu', torch.zeros([args.ext_size]))
        self.register_buffer('prior_ext_logvar', torch.zeros([args.ext_size]))
        self.register_buffer('prior_bck_mu', torch.zeros([args.bck_size]))
        self.register_buffer('prior_bck_logvar', torch.zeros([args.bck_size]))


    def forward(self, video, tau):
        B, C, H, W = video.size()

        with torch.no_grad():
            # Pretrained Dino feature extractor
            dino_feat = self.dino_feature_extractor(video*255)
            feat = np.stack(dino_feat['pixel_values'])
            out_dino = self.dino_encoder(torch.tensor(feat).cuda()).detach()               # [B, 768, 28, 28]
            feat_sce = out_dino
        
        if self.args.data_name == 'clevr' or self.args.data_name == 'shop' or self.args.data_name == 'octa':
            # Encode feature for scene image
            feat_sce = self.ocl_encoder.cnn(video)           # B, d_dino, H, W
            feat_sce = self.ocl_encoder.pos(feat_sce)             # B, d_dino, H, W


        # Encode background
        slots_bck = self.ocl_encoder.enc_bck(feat_sce.flatten(1)).unsqueeze(1)           # [B, 1, bck_size]

        # Encode feature for Slot Attention
        H_enc, W_enc = feat_sce.shape[-2:]
        emb_set = feat_sce.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)             # B, H * W, d_dino
        emb_set = self.ocl_encoder.mlp(self.ocl_encoder.layer_norm(emb_set))           # B, H * W, d_dino

        # Slot Attention
        slots_bck_att = self.ocl_encoder.bck_proj(slots_bck)                                               # slots: B, 1, slot_size
        (slots_ext, logits_pi, attns) = self.ocl_encoder.savi(slots_bck_att, emb_set, self.ocl_encoder.protos, tau)    # slots: B, num_slots, slot_size                                                     
        attns = attns.transpose(-1, -2).reshape(B, self.num_slots, 1, H_enc, W_enc)                          # B, num_slots, 1, H_enc, W_enc
        
        pi = F.gumbel_softmax(logits_pi, tau=tau, hard=False)
        pi_c = torch.argmax(pi, dim=-1)
        slots_int = (pi[:,:,:, None] * self.ocl_encoder.protos[None,None]).sum(-2)


        # Reparameters-trick
        ext_mu, ext_logvar = self.ocl_encoder.enc_ext(slots_ext).chunk(2,-1)
        slots_ext = reparameterize_normal(ext_mu, ext_logvar)

        # Reparameters-trick
        bck_mu, bck_logvar = self.ocl_encoder.bck2lat(slots_bck).chunk(2,-1)
        slots_bck = reparameterize_normal(bck_mu, bck_logvar)

    
        

        # decoded_patches
        slots_obj = torch.cat([slots_int, slots_ext], dim=-1)
        slots_obj = self.ocl_decoder.slot_proj(slots_obj)                                          # B, num_slots, d_vqvae
        slots_obj = slots_obj.flatten(0, -2)                                                         # B * num_slots, d_vqvae
        slots_obj = slots_obj.unsqueeze(1).expand(-1, self.num_dino_patches, -1)                     # B  num_slots, num_patches, d_vqvae
        # Simple learned additive embedding as in ViT
        slots_obj = slots_obj + self.ocl_decoder.pos_embed
        
        output_obj = self.ocl_decoder.decoder_obj(slots_obj)                                       # B * num_slots, num_patches, d_vqvae + 1
        output_obj = output_obj.reshape(B, self.num_slots-1, self.num_dino_patches, -1)              # B, num_slots, num_patches, d_vqvae + 1
        # Split out alpha channel and normalize over slots.
        decoded_patches_obj, alpha_obj = output_obj.split([self.d_dino, 1], dim=-1)                  # B , num_slots, num_patches, d_vqvae / 1
        
        # decoder background
        output_bck = self.ocl_decoder.decoder_bck(slots_bck).reshape(B, 1, self.num_dino_patches, self.d_dino+1)
        decoded_patches_bck, alpha_bck = output_bck.split([self.d_dino, 1], dim=-1)   

        # normalize mask over slots
        alpha = torch.cat([alpha_bck, alpha_obj], dim=-3)
        alpha = alpha.softmax(dim=-3)                                                           # B, num_slots, num_patches, 1
        masks = alpha.transpose(-1, -2).reshape(B, self.num_slots, 1, H_enc, W_enc)                 # B, num_slots, 1, H_enc, W_enc

        decoded_patches = torch.cat([decoded_patches_bck, decoded_patches_obj], dim=-3)
        mix_patch = torch.sum(decoded_patches * alpha, dim=-3)                                  # B, num_patches, d_vqvae
        
        # Calculate Loss
        # Reg Background 
        reg_bck = ((out_dino.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).detach() - decoded_patches_bck.squeeze(1)) **2).sum() / B
        # Patch Loss
        mse = (( out_dino.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).detach() - mix_patch) ** 2).sum() / B                     # 1


        kld_ext = compute_kld_normal(ext_mu, ext_logvar, self.prior_ext_mu, self.prior_ext_logvar).sum() / B
        kld_bck = compute_kld_normal(bck_mu, bck_logvar, self.prior_bck_mu, self.prior_bck_logvar).sum() / B
        post_gamma = Categorical(logits=logits_pi)
        prior_gamma = Categorical(self.prior_logits_pi)
        kld_pi= kl_divergence(post_gamma, prior_gamma.expand(post_gamma.batch_shape)).sum()  / B

        klds = kld_ext + kld_bck + kld_pi


        
        return (mse, klds, reg_bck, attns, masks, pi_c)

    def encode(self, video, tau):
        B, C, H, W = video.size()

        with torch.no_grad():
            # Pretrained Dino feature extractor
            dino_feat = self.dino_feature_extractor(video*255)
            feat = np.stack(dino_feat['pixel_values'])
            out_dino = self.dino_encoder(torch.tensor(feat).cuda()).detach()               # [B, 768, 28, 28]
            feat_sce = out_dino
        
        if self.args.data_name == 'clevr' or self.args.data_name == 'shop' or self.args.data_name == 'octa':
            # Encode feature for scene image
            feat_sce = self.ocl_encoder.cnn(video)           # B, d_dino, H, W
            feat_sce = self.ocl_encoder.pos(feat_sce)             # B, d_dino, H, W


        # Encode background
        slots_bck = self.ocl_encoder.enc_bck(feat_sce.flatten(1)).unsqueeze(1)           # [B, 1, bck_size]

        # Encode feature for Slot Attention
        H_enc, W_enc = feat_sce.shape[-2:]
        emb_set = feat_sce.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)             # B, H * W, d_dino
        emb_set = self.ocl_encoder.mlp(self.ocl_encoder.layer_norm(emb_set))           # B, H * W, d_dino

        # Slot Attention
        slots_bck_att = self.ocl_encoder.bck_proj(slots_bck)                                               # slots: B, 1, slot_size
        (slots_ext, logits_pi, attns) = self.ocl_encoder.savi(slots_bck_att, emb_set, self.ocl_encoder.protos, tau)    # slots: B, num_slots, slot_size                                                     
        attns = attns.transpose(-1, -2).reshape(B, self.num_slots, 1, H_enc, W_enc)                          # B, num_slots, 1, H_enc, W_enc
        
        pi = F.gumbel_softmax(logits_pi, tau=tau, hard=False)
        pi_c = torch.argmax(pi, dim=-1)
        slots_int = (pi[:,:,:, None] * self.ocl_encoder.protos[None,None]).sum(-2)


        # Reparameters-trick
        ext_mu, ext_logvar = self.ocl_encoder.enc_ext(slots_ext).chunk(2,-1)
        slots_ext = reparameterize_normal(ext_mu, ext_logvar)

        # Reparameters-trick
        bck_mu, bck_logvar = self.ocl_encoder.bck2lat(slots_bck).chunk(2,-1)
        slots_bck = reparameterize_normal(bck_mu, bck_logvar)


        return (slots_int, slots_ext, slots_bck, pi, attns)

    def objdec(self, slots_int, slots_ext):
        B, K, _ = slots_ext.size()

        # decoded_patches
        slots_obj = torch.cat([slots_int, slots_ext], dim=-1)
        slots_obj = self.ocl_decoder.slot_proj(slots_obj)                                          # B, num_slots, d_dino
        slots_obj = slots_obj.flatten(0, -2)                                                         # B * num_slots, d_dino
        slots_obj = slots_obj.unsqueeze(1).expand(-1, self.num_dino_patches, -1)                     # B * num_slots, num_patches, d_dino
        # Simple learned additive embedding as in ViT
        slots_obj = slots_obj + self.ocl_decoder.pos_embed
        output_obj = self.ocl_decoder.decoder_obj(slots_obj)                                       # B * num_slots, num_patches, d_dino + 1
        output_obj = output_obj.reshape(B, K, self.num_dino_patches, -1)              # B , num_slots, num_patches, d_dino + 1
        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output_obj.split([self.d_dino, 1], dim=-1)                          # B, num_slots, num_patches, d_dino / 1
        # alpha = torch.sigmoid(alpha)                                                                # B, num_slots, num_patches, 1

        # patches = decoded_patches * alpha
        
        # masks = alpha.transpose(-1, -2).reshape(B, K, 1, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))                   # B, num_slots, 1, H_enc, W_enc
        # masks_vis = torch.nn.functional.interpolate(
        #     masks.flatten(end_dim=1), 
        #     size=(self.image_size, self.image_size),
        #     mode='bilinear'
        #     )   # [B*K, 1, H, W]
        # masks_vis = masks_vis.reshape(*masks.shape[:2], 1, self.image_size, self.image_size)

        return decoded_patches, alpha


    def visual_dis(self, video, tau):
        (slots_int, slots_ext, _,_,_) = self.encode(video, tau)
        
        int1 = slots_int[:, :1]
        ext1 = slots_ext[:, :1]
        int2 = slots_int[:, 1:2]
        ext2 = slots_ext[:, 1:2]

        _, mask1 = self.objdec(int1, ext1)
        _, mask2 = self.objdec(int2, ext2)
        _, mask_ex12 = self.objdec(int1, ext2)
        _, mask_ex21 = self.objdec(int2, ext1)
        mask_exc = torch.cat([mask1[:,0], mask_ex12[:,0], mask2[:,0], mask_ex21[:,0]])  #[B*4,1,H,W]
        mask_exc = mask_exc.permute(0,2,1).reshape(-1, 1, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))
        

        randn_ext = torch.randn(4, self.ext_size).cuda()
        rad_ext = randn_ext[:,None].expand(-1, self.num_cls, -1).reshape(-1, self.ext_size)
        rad_protos = self.ocl_encoder.protos[None].expand(randn_ext.shape[0], -1, -1).reshape(-1, self.int_size)
        _, mask_pro = self.objdec(rad_protos[:,None], rad_ext[:,None])
        mask_pro = mask_pro.squeeze(1)  # 4*10, 1, H, W
        mask_pro = mask_pro.permute(0,2,1).reshape(-1, 1, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))

        return mask_exc, mask_pro


    def decode(self, slots_int, slots_ext, slots_bck):
        B,_,_ = slots_int.size()
        # decoded_patches
        slots_obj = torch.cat([slots_int, slots_ext], dim=-1)
        slots_obj = self.ocl_decoder.slot_proj(slots_obj)                                          # B, num_slots, d_vqvae
        slots_obj = slots_obj.flatten(0, -2)                                                         # B * num_slots, d_vqvae
        slots_obj = slots_obj.unsqueeze(1).expand(-1, self.num_dino_patches, -1)                     # B  num_slots, num_patches, d_vqvae
        # Simple learned additive embedding as in ViT
        slots_obj = slots_obj + self.ocl_decoder.pos_embed
        
        output_obj = self.ocl_decoder.decoder_obj(slots_obj)                                       # B * num_slots, num_patches, d_vqvae + 1
        output_obj = output_obj.reshape(B, self.num_slots-1, self.num_dino_patches, -1)              # B, num_slots, num_patches, d_vqvae + 1
        # Split out alpha channel and normalize over slots.
        decoded_patches_obj, alpha_obj = output_obj.split([self.d_dino, 1], dim=-1)                  # B , num_slots, num_patches, d_vqvae / 1
        
        # decoder background
        output_bck = self.ocl_decoder.decoder_bck(slots_bck).reshape(B, 1, self.num_dino_patches, self.d_dino+1)
        decoded_patches_bck, alpha_bck = output_bck.split([self.d_dino, 1], dim=-1)   

        # normalize mask over slots
        alpha = torch.cat([alpha_bck, alpha_obj], dim=-3)
        alpha = alpha.softmax(dim=-3)                                                           # B, num_slots, num_patches, 1
        masks = alpha.transpose(-1, -2).reshape(B, self.num_slots, 1, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))                 # B, num_slots, 1, H_enc, W_enc

        decoded_patches = torch.cat([decoded_patches_bck, decoded_patches_obj], dim=-3)
        patches_all = decoded_patches * alpha                                  # B, K+1, num_patches, d_vqvae

        return patches_all, masks


    def dec_bck(self, slots_bck):
        # decoder background
        output_bck = self.ocl_decoder.decoder_bck(slots_bck).reshape( slots_bck.shape[0], 1, self.num_dino_patches, self.d_dino+1)
        decoded_patches_bck, alpha_bck = output_bck.split([self.d_dino, 1], dim=-1) 

        # masks = alpha_bck.transpose(-1, -2).reshape(1, 1, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))                   # B, num_slots, 1, H_enc, W_enc
        # masks_vis = torch.nn.functional.interpolate(
        #     masks, 
        #     size=(self.image_size, self.image_size),
        #     mode='bilinear'
        #     )   # [B*K, 1, H, W]
        # bck_mask = masks_vis.reshape(*masks.shape[:1], 1, self.image_size, self.image_size)

        return decoded_patches_bck, alpha_bck