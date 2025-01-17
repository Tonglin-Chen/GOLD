import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# from nerv.training import BaseModel

from .modules import Decoder
from .quantize import VectorQuantizer2 as VectorQuantizer
from .loss import VQLPIPSLoss
import pdb

def temporal_wrapper(func):
    """A wrapper to make the model compatible with both 4D and 5D inputs."""

    def f(cls, x):
        """x is either [B, C, H, W] or [B, T, C, H, W]."""
        B = x.shape[0]
        if len(x.shape) == 5:
            unflatten = True
            x = x.flatten(0, 1)
        else:
            unflatten = False

        outs = func(cls, x)

        if unflatten:
            if isinstance(outs, tuple):
                outs = [o.unflatten(0, (B, -1)) if o.ndim else o for o in outs]
                return tuple(outs)
            else:
                return outs.unflatten(0, (B, -1))
        else:
            return outs

    return f


class VQVAE1(nn.Module):
    """VQ-VAE consisting of Encoder, QuantizationLayer and Decoder."""

    def __init__(
        self,
        enc_dec_dict=dict(
            resolution=128,
            in_channels=3,
            z_channels=3,
            ch=64,
            ch_mult=[1, 2, 4],  # num_down = len(ch_mult)-1
            num_res_blocks=2,
            attn_resolutions=[],
            out_ch=3,
            dropout=0.0,
        ),
        vq_dict=dict(
            n_embed=4096,  # vocab_size
            embed_dim=3,  # same as `z_channels`
            percept_loss_w=1.0,
        ),
        use_loss=True,
    ):
        super().__init__()

        self.resolution = enc_dec_dict['resolution']
        self.embed_dim = vq_dict['embed_dim']
        self.n_embed = vq_dict['n_embed']
        self.z_ch = enc_dec_dict['z_channels']

        # self.encoder = Encoder(**enc_dec_dict)
        self.decoder = Decoder(**enc_dec_dict)

        self.quantize = VectorQuantizer(
            self.n_embed,
            self.embed_dim,
            beta=0.25,
            sane_index_shape=True,
        )

        self.post_quant_conv = nn.Conv2d(self.embed_dim, self.z_ch, 1)

        if use_loss:
            self.loss = VQLPIPSLoss(percept_loss_w=vq_dict['percept_loss_w'])


    def calc_train_loss(self, img, out_dict):
        """Compute training loss."""
        recon = out_dict['recon']
        quant_loss = out_dict['quant_loss']

        loss_dict = self.loss(quant_loss, img, recon)

        return loss_dict


