# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import torch.nn.functional as F


def morph(src_bg_mask, ks, mode="erode", kernel=None):
    """

    Args:
        src_bg_mask (torch.tensor):
        ks (int):
        mode (str):
        kernel (torch.tensor or None):

    Returns:
        out (torch.tensor):

    """
    n_ks = ks ** 2
    pad_s = ks // 2

    if kernel is None:
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32, device=src_bg_mask.device)

    if mode == "erode": # erode the mask.
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=1.0)
        out = F.conv2d(src_bg_mask_pad, kernel)
        out = (out == n_ks).float()
    else: # dilate the mask
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=0.0)
        out = F.conv2d(src_bg_mask_pad, kernel)
        out = (out >= 1).float()

    return out


def soft_dilate(src_bg_mask, ks, kernel=None):
    """

    Args:
        src_bg_mask (torch.tensor):
        ks (int):
        kernel (torch.tensor or None):

    Returns:
        out (torch.tensor):

    """
    n_ks = ks ** 2
    pad_s = ks // 2

    if kernel is None:
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32, device=src_bg_mask.device)

    src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=0.0)
    out = F.conv2d(src_bg_mask_pad, kernel)
    out = (out >= n_ks / 2).float()

    return out

if __name__=='__main__':
    bg = torch.ones((2,1,512,512))
    ks = 13
    b = morph(bg,ks)
