import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .SurfaceClassifier import conv1_1, period_loss
# from .DepthNormalizer import DepthNormalizer
from ..net_util import *
# from iPERCore.models.networks.criterions import VGGLoss
from lib.model.Models import NestedUNet
import numpy as np


class Pint_Model(nn.Module):
    def __init__(self, opt):
        super(Pint_Model, self).__init__()
        self.period_loss = period_loss()
        self.feat_uv_error = nn.SmoothL1Loss() # A feature with B uvmap
        self.opt = opt
        self.NUnet = NestedUNet(in_ch=3, out_ch=3)
        norm_type = get_norm_layer(norm_type=opt.norm_color)
        self.image_filter = ResnetFilter(opt, norm_layer=norm_type)
        # self.conv = conv1_1(input_layers=256, output_layers=16)
        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat = self.image_filter(images)

    def forward(self, uv_A, uv_B, part_uv_B, index):
        '''
        this function is made for pint total train.
        '''
        complete_feat = self.NUnet(uv_A)
        complete_feat_B = self.NUnet(uv_B)
        
        # im_feat = self.image_filter(uv_A) # B C H W for 512 uv_B, B 256 128 128
        # complete_feat = self.conv(im_feat) # B 16 128 128 -> b 16 512 512 [0:3] loss

        # im_feat_B = self.image_filter(uv_B)
        # complete_feat_B = self.conv(im_feat_B)
        # A_feat = F.interpolate(complete_feat[:,0:3,:,:], scale_factor=4, mode='bilinear', align_corners=True) # in this param, A_feat means complete feature.

        # part_uv_B.requires_grad=True # to make uvb as one leaf
        # A_feat = complete_feat[:,0:3,:,:]
        # part_uv_B = F.interpolate(part_uv_B, scale_factor=0.25, mode='bilinear', align_corners=True)
        
        A_vis_feat = complete_feat[index==1]
        B_vis_uv = part_uv_B[index==1]

        loss1 = self.feat_uv_error(A_vis_feat, B_vis_uv.detach())
        # loss2 = self.vgg_loss(complete_feat[:,:3], complete_feat_B[:,:3].detach())
        # loss2 = self.period_loss(complete_feat, complete_feat_B.detach())
        loss2=0
        return complete_feat, complete_feat_B, loss1, loss2

    # def pint_forward(self, uv_A, uv_B):
    #     '''
    #     this function is made for pint total train.
    #     '''
    #     im_feat = self.image_filter(uv_A) # B C H W for 512 uv_B, B 256 128 128
    #     self.complete_feat = self.conv(im_feat) # B 16 128 128 -> b 16 512 512 [0:3] loss

    #     im_feat_B = self.image_filter(uv_B.squeeze(1))
    #     complete_feat_B = self.conv(im_feat_B)
    #     A_feat = F.interpolate(self.complete_feat[:,0:3,:,:], scale_factor=4, mode='bilinear', align_corners=True) # in this param, A_feat means complete feature.
    #     uv_B_feat = uv_B.squeeze(1).expand_as(A_feat)
    #     uv_B_feat.requires_grad=True # to make uvb as one leaf
    #     A_vis_feat = A_feat[uv_B_feat != 0.0]
    #     B_vis_uv = uv_B_feat[uv_B_feat != 0.0]
    #     loss_content = self.feat_uv_error(A_vis_feat, B_vis_uv) * 100
    #     loss_content1 = self.feat_uv_error(A_feat, uv_A)*100
    #     # loss_feat = self.error_term(self.complete_feat, complete_feat_B)
    #     return A_feat, A_vis_feat, B_vis_uv, self.complete_feat, complete_feat_B, loss_content+loss_content1

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, last)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if last:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ResnetFilter(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, opt, input_nc=3, output_nc=256, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetFilter, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            if i == n_blocks - 1:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias, last=True)]
            else:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)]

        if opt.use_tanh:
            model += [nn.Tanh()]
    
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
