import json
import random
import os.path
import argparse
from datetime import datetime

import torch
import torchvision.utils as vutils

import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import lpips
from torchvision.utils import save_image

import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from plot_decompose import *



def compute_ari_ami(segment, masks):
    segment = segment.cpu().numpy()
    masks = masks.cpu().numpy()
    segment_true = segment
    segment_sel = segment >= 1
    segment_a = masks
    outputs = {key: [] for key in ['ari_all', 'ari_obj', 'ami_all', 'ami_obj']}
    for seg_true, seg_sel, seg_a in zip(segment_true, segment_sel, segment_a):
        seg_a_true_sel = seg_true.reshape(-1)
        seg_o_true_sel = seg_true[seg_sel]
        seg_a_sel = seg_a.reshape(-1)
        seg_o_sel = seg_a[seg_sel]
        outputs['ari_all'].append(adjusted_rand_score(seg_a_true_sel, seg_a_sel))
        outputs['ari_obj'].append(adjusted_rand_score(seg_o_true_sel, seg_o_sel))
        outputs['ami_all'].append(
            adjusted_mutual_info_score(seg_a_true_sel, seg_a_sel, average_method='arithmetic'))
        outputs['ami_obj'].append(
            adjusted_mutual_info_score(seg_o_true_sel, seg_o_sel, average_method='arithmetic'))
    outputs = {key: np.array(val).mean(-1) for key, val in outputs.items()}
    return outputs


def select_by_index(x, index_raw):
    x = torch.from_numpy(x)
    index = torch.from_numpy(index_raw)
    x_ndim = x.ndim
    index_ndim = index.ndim
    index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
    index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
    if index_raw.ndim == 2:
        x = torch.gather(x, index_ndim - 1, index)
    elif index_raw.ndim == 3:
        x = torch.gather(x, index_ndim - 1, index)
    elif index_raw.ndim == 4:
        x = torch.gather(x, index_ndim - 1, index)
    else:
        raise AssertionError
    return x.numpy()


def compute_iou_f1(segment, masks_pre, eps=1e-6):
    segment = segment[:, None] # (B, 1, H, W)
    scatter_shape = [*segment.shape[:1], 7, *segment.shape[2:]] # (B, K1, H, W)
    # scatter_shape = [*segment.shape[:1], (segment.max() + 1).item(), *segment.shape[2:]] # (B, K1, H, W)
    segment = segment.cpu()
    
    masks_true = torch.zeros(scatter_shape).scatter_(1, segment, 1).numpy().astype(np.float64) # (B, K1, H, W)
    masks = masks_pre[:, None] # (B, 1, H, W)
    scatter_shape = [*masks.shape[:1], 8, *masks.shape[2:]] # (B, K2, H, W)
    # scatter_shape = [*masks.shape[:1], (masks.max() + 1).item(), *masks.shape[2:]] # (B, K2, H, W)
    masks = masks.cpu()
    masks = torch.zeros(scatter_shape).scatter_(1, masks, 1).numpy().astype(np.float64) # (B, K2, H, W)
    order_cost_all = -(masks_true[:, :, None] * masks[:, None]) # (B, K1, K2, H, W)
    order_cost_all = order_cost_all.reshape(*order_cost_all.shape[:-2], -1).sum(-1) # (B, K1, K2)
    order_all = []
    for cost in order_cost_all:
        _, cols = linear_sum_assignment(cost) # (K1)
        order_all.append(cols)
    order_all = np.array(order_all) # (B, K1)
    masks = select_by_index(masks, order_all) # (B, K2, H, W)
    seg_true = masks_true.reshape(*masks_true.shape[:2], -1) # (B, K1, HxW)
    pres = (seg_true.max(-1) != 0).astype(np.float64) # (B, K1)
    sum_pres = pres.sum()
    seg_pred = masks.reshape(*masks.shape[:2], -1) # (B, K1, HxW)
    
    area_i = np.minimum(seg_true, seg_pred).sum(-1) # (B, K1)
    area_u = np.maximum(seg_true, seg_pred).sum(-1) # (B, K1)
    iou = area_i / np.clip(area_u, eps, None)
    f1 = 2 * area_i / np.clip(area_i + area_u, eps, None)
    iou = (iou * pres).sum() / sum_pres
    f1 = (f1 * pres).sum() / sum_pres
    return iou, f1


def select_by_index_acc(x, index):
    x_ndim = x.ndim
    index_ndim = index.ndim
    index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
    index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
    x = torch.gather(x, index_ndim - 1, index)
    return x


def compute_pairs(video, segment, mask_pre, clss_true, clss_pred):

    segment_base = segment[:, None, None].long()
    scatter_shape = [*segment_base.shape[:1], 7, *segment_base.shape[2:]]
    mask_true = torch.zeros(scatter_shape, device=segment_base.device).scatter_(1, segment_base, 1)   # [B, K, 1, H, W]

    mask_true = mask_true[:,1:]
    obj_true  = video.unsqueeze(1) * mask_true    # [B, K, C, H, W]

    obj_pre = video.unsqueeze(1) * mask_pre

    #(B,C=num_object) C in [-1~9],11 kinds of object, -1 means black, no object
    pairs_batch = torch.LongTensor(*clss_true.shape,2).fill_(0).cuda()#(B,C,2) last dim 2 corrospond to label_true and label_pred respectively
    # wy add ending

    #match the order
    costs = torch.sqrt((obj_true.unsqueeze(2) - obj_pre.unsqueeze(1)).pow(2).sum((-3,-2,-1))).cpu().detach().numpy()   

    # print('costs.shape:',costs.shape)#(128,3,3)
    for s in range(len(costs)):#len=B
        rows,cols = linear_sum_assignment(costs[s])#rows and cols in range[0~2], so in wy_added, we change it to label
        pairs_batch[s, :, 0] = clss_true[s][rows]
        pairs_batch[s, :, 1] = clss_pred[s][cols]

    return pairs_batch

def cluster_acc(num_cls, pairs_all):
    # compute clustering accuracy
    # - matrix weight
    cls_acc_mtx = torch.LongTensor(num_cls+1,num_cls+1).fill_(0).cuda()

    for pairs in pairs_all:#paris:(B,C,2)
        pairs[pairs<0] = num_cls #change label -1 to 10 for convinient computing
        for i_p_s in range(pairs.shape[0]):
            # cls_acc_mtx[pairs[i_p_s,:,0], pairs[i_p_s,:,1]] += 1#every label_pair add 1 to matrix ## vector method will mute same index, rendered incorrect count
            for i_p_s_2 in range(pairs.shape[1]):
                cls_acc_mtx[pairs[i_p_s, i_p_s_2, 0], pairs[i_p_s, i_p_s_2, 1]] += 1
    # - match
    def _make_cost_m(cm):
        s = torch.max(cm)
        return (- cm + s)

    matches = linear_sum_assignment(_make_cost_m(cls_acc_mtx[:num_cls,:num_cls]).cpu())
    clustering_acc = (torch.sum(cls_acc_mtx[matches[0],matches[1]]) + torch.sum(cls_acc_mtx[-1])) / (torch.sum(cls_acc_mtx)+.0)

    return clustering_acc




def visualize_seg_overlay_image(image, mask, batch, save_path):
    B,  C, H, W = image.shape
    image = image.reshape(B, C, H, W).cpu()

    mask = mask.squeeze(2)          # [B, K, 1, H, W] -> [B, K, H, W]
    mask = mask.cpu().contiguous()  # B, K, H, W
    n_objects = mask.shape[1]

    masks_argmax = mask.argmax(dim=1)[:, None]  # B, 1, H, W
    classes = torch.arange(n_objects)[None, :, None, None].to(masks_argmax) # 1, K, 1, 1
    masks_one_hot = masks_argmax == classes # B, K, H, W

    from matplotlib import cm
    mpl_cmap = cm.get_cmap("tab20", n_objects)(range(n_objects))
    cmap = [tuple((255 * cl[:3]).astype(int)) for cl in mpl_cmap]
    
  
    masks_on_image = torch.stack(
        [
            draw_segmentation_masks(
                (255 * img).to(torch.uint8), mask, alpha=0.75, colors=cmap
            )
            for img, mask in zip(image.to("cpu"), masks_one_hot.to("cpu"))
        ]
    )
    masks_on_image = masks_on_image.reshape(B, C, H, W) / 255.

    com = torch.cat([image, masks_on_image])

    vutils.save_image(com, os.path.join(save_path, 'OIGO_img_overlay_mask_{}.png'.format(batch)), nrow=image.shape[0], padding=2, pad_value=1)



def draw_segmentation_masks(image, masks, alpha=0.8, colors=None):
    """
    Draws segmentation masks on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (color or list of colors, optional): List containing the colors
            of the masks or single color for all masks. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")

    num_masks = masks.size()[0]
    if colors is not None and num_masks > len(colors):
        raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

    if num_masks == 0:
        print("masks doesn't contain any mask. No mask was drawn")
        return image

    if not isinstance(colors, list):
        colors = [colors]
    if not isinstance(colors[0], (tuple, str)):
        raise ValueError("colors must be a tuple or a string, or a list thereof")
    if isinstance(colors[0], tuple) and len(colors[0]) != 3:
        raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

    out_dtype = torch.uint8

    from PIL import ImageColor

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        colors_.append(torch.tensor(color, dtype=out_dtype))

    img_to_draw = image.detach().clone()
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)



def visualize_decompose(image, masks, masks_vis, batch, save_path):
    B, K, C, H, W = masks_vis.shape
    
    mask = masks.reshape(B, K, 1, H, W)
    mask_oh = torch.argmax(mask, dim=1, keepdim=True)
    mask_oh = torch.zeros_like(mask).scatter_(1, mask_oh, 1)
    pres = mask_oh.reshape(*mask_oh.shape[:-3], -1).max(-1).values  #[B, K]

    fig = plot_decompose(B,image, mask, masks_vis, pres)
    fig.savefig(os.path.join(save_path, 'OIGO_decompose_{}.png'.format(batch)), bbox_inches='tight', pad_inches=0)
    plt.close()
        
    return 

