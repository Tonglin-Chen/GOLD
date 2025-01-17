import argparse
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib.colors import hsv_to_rgb
import pdb
plt.rc('font', size=32, family='serif')
# plt.rc('text', usetex=True)


def compute_segre_combine(segre):
    num_colors = segre.shape[0]
    hsv_colors = np.ones((num_colors, 3))
    hsv_colors[:, 0] = (np.linspace(0, 1, num_colors, endpoint=False) + 2 / 3) % 1.0
    segre_colors = hsv_to_rgb(hsv_colors)
    segre_combine = np.clip((segre * segre_colors[:, None, None]).sum(0), 0, 1)


    # ins_seg = torch.argmax(torch.tensor(segre.squeeze(-1)),0,True)  # [1, H, W]
    # ins_seg = ins_seg.cpu().numpy().transpose(1,2,0)                # [H, W, 1] 
    # img_r = np.zeros_like(ins_seg)
    # img_g = np.zeros_like(ins_seg)
    # img_b = np.zeros_like(ins_seg)
    # for c_idx in range(ins_seg.max().item() + 1):
    #     c_map = ins_seg == c_idx
    #     if c_map.any():
    #         img_r[c_map] = segre_colors[c_idx][0]
    #         img_g[c_map] = segre_colors[c_idx][1]
    #         img_b[c_map] = segre_colors[c_idx][2]
    # segre_combine = np.concatenate([img_r, img_g, img_b], axis=-1)

    return segre_combine, segre_colors


def color_spines(ax, color, lw=3):
    for loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[loc].set_linewidth(lw)
        ax.spines[loc].set_color(color)
        ax.spines[loc].set_visible(True)
    return


def plot_image(ax, image, xlabel=None, ylabel=None, border_color=None):
    ax.imshow(image, interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color='k') if xlabel else None
    ax.set_ylabel(ylabel, color='k') if ylabel else None
    ax.xaxis.set_label_position('top')
    if border_color:
        color_spines(ax, color=border_color)
    return


def plot_null(ax, xlabel=None, ylabel=None):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color='k') if xlabel else None
    ax.set_ylabel(ylabel, color='k') if ylabel else None
    ax.xaxis.set_label_position('top')
    color_spines(ax, color=None)
    return


# def select_by_index(x, index_raw):
#     x = torch.from_numpy(x)
#     index = torch.from_numpy(index_raw)
#     x_ndim = x.ndim
#     index_ndim = index.ndim
#     index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
#     index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
#     if index_raw.ndim == 2:
#         x_obj = torch.gather(x[:, :-1], index_ndim - 1, index)
#         x = torch.cat([x_obj, x[:, -1:]], dim=1)
#     elif index_raw.ndim == 3:
#         x_obj = torch.gather(x[:, :, :-1], index_ndim - 1, index)
#         x = torch.cat([x_obj, x[:, :, -1:]], dim=2)
#     else:
#         raise AssertionError
#     return x.numpy()



# def select_by_index_acc(x, index):
#     x_ndim = x.ndim
#     index_ndim = index.ndim
#     index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
#     index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
#     x = torch.gather(x, index_ndim - 1, index)
#     return x


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



def plot_decompose(num_data, image, masks, apc, pres, scale=2):

    image = image.data.cpu().numpy()      # B, C, H , W
    images = np.moveaxis(image, -3, -1)
    mask = masks.data.cpu().numpy()
    mask = np.moveaxis(mask, -3, -1)      # B,K+1, 1,H, W
    apc = apc.data.cpu().numpy()
    apc = np.moveaxis(apc, -3, -1)        # B,K+1, C,H, W
    pres = pres.data.cpu().numpy()        # B,K+1

    mask_b = mask[:,:1]
    mask_o = mask[:,1:]   # B,K, 1,H, W
    apc_b = apc[:,:1]    
    apc_o = apc[:,1:]     # B,K, C,H, W
    pres_b = pres[:,:1]
    pres_o = pres[:,1:]  # B,K


    order = np.argsort(-pres_o, axis=-1)
    mask_o = select_by_index(mask_o, order)
    apc_o = select_by_index(apc_o, order)
    pres_o = select_by_index(pres_o, order)

    # Last slot is background
    mask_all = np.concatenate([mask_o, mask_b], axis=1)
    apc_all = np.concatenate([apc_o, apc_b], axis=1)
    pres_all = np.concatenate([pres_o, pres_b], axis=1)

    obj_slots = apc_all.shape[1]
    num_cols = num_data
    num_rows = obj_slots + 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * scale, num_rows * scale))
    idx_data = 0
    for idx_view in range(num_data):
        col = idx_view

        segre_combine, segre_colors = compute_segre_combine(mask_all[idx_view])
        plot_image(axes[0, col], images[idx_view], ylabel='image' if col == 0 else None)
        plot_image(axes[1, col], segre_combine, ylabel='seg' if col == 0 else None)
        
        row = 2
        for idx_obj in range(apc_all.shape[1]):
            border_color = tuple(segre_colors[idx_obj])
            if idx_obj < apc_all.shape[1] - 1:
                ylabel = 'obj {}'.format(idx_obj + 1)  
            else: 
                ylabel = 'bck'
            plot_image(axes[row, col], apc_all[idx_view, idx_obj], border_color=border_color, ylabel=ylabel if col == 0 else None)
            row += 1
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig



