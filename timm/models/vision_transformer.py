""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2021 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .helpers import build_model_with_cfg, named_apply, adapt_input_conv
from .layers import PatchEmbed, Mlp, ConvMlpGeneral, DropPath, trunc_normal_, lecun_normal_
from .registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}

########################################################################################################################

class Attention(nn.Module):
    def __init__(self, dim, dim_ratio=1, act=False, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., depth=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = (dim * 1) // num_heads
        self.scale = head_dim ** -0.5
        self.dim_ratio = dim_ratio
        self.act = nn.GELU() if act else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 2 + dim * dim_ratio * 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim * dim_ratio, dim)
        #self.proj2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #self.dense_mix = nn.Linear(min(depth+1,3) * dim, dim, bias=qkv_bias) if depth>0 else None
        '''self.dense_mul = min(depth+1,3)
        #self.dense_prob = [[1.], [0.6, 0.4], [0.5, 0.3, 0.2]][self.dense_mul-1]
        #self.dense_test_ind=[[0]*10, [0,1,0,1,0]*2, [0,0,1,0,2,1,0,0,1,2]][self.dense_mul-1]

        #MAX_DEPTH = 12-1
        #p_self = 0.5 + 0.4 * abs(2*depth/MAX_DEPTH - 1)
        #self.dense_prob = [[1.], [p_self, 1-p_self], [p_self, (1-p_self)*0.75, (1-p_self)*0.25]][self.dense_mul-1]

        self.dense_mix_q = nn.Parameter(torch.FloatTensor(197, min(depth+1,3))) if depth>0 else None
        self.dense_mix_k = nn.Parameter(torch.FloatTensor(197, min(depth+1,3))) if depth>0 else None
        self.dense_mix_v = nn.Parameter(torch.FloatTensor(197, min(depth+1,3))) if depth>0 else None'''


    def forward(self, x):#, prev_q, prev_k, prev_v):
        B, N, C = x.shape

        '''#res = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        qkv = self.qkv(x).reshape(B, N, 1 + dim_ratio*2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B h N C//h'''

        qkv = self.qkv(x)
        #v, q, k = qkv[...,:C], qkv[...,C:(1+self.dim_ratio)*C], qkv[...,(1+self.dim_ratio)*C:] C 2C=q 2C=k
        q, k, v = qkv[...,:C], qkv[...,C:2*C], qkv[...,2*C:] # C C 2C=v
        q = q.reshape(B, N, self.num_heads, -1).permute(0,2,1,3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0,2,1,3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0,2,1,3)

        '''q += res
        k += res
        v += res'''

        #q0, k0, v0 = q, k, v
        '''
        if prev_k is not None:
            #k, v = torch.cat([k, prev_k], dim=2), torch.cat([v, prev_v], dim=2) # B h N 3C//h
            k, v = torch.cat([k, prev_k], dim=-1), torch.cat([v, prev_v], dim=-1)
            k, v = rearrange(k, 'b h n c -> b n (h c)'), rearrange(v, 'b h n c -> b n (h c)')
            #print(k.shape, self.dense_mix.weight.shape)
            k, v = self.dense_mix(k), self.dense_mix(v)
            k = rearrange(k, 'b n (h c) -> b h n c', h=self.num_heads, c=C // self.num_heads)
            v = rearrange(v, 'b n (h c) -> b h n c', h=self.num_heads, c=C // self.num_heads)

        if prev_k is not None:
            k, v = torch.cat([k, prev_k], dim=-2), torch.cat([v, prev_v], dim=-2) # B h 3N C//h

            #if self.training:
            k, v = rearrange(k, 'b h n c -> (b n) h c'), rearrange(v, 'b h n c -> (b n) h c')
            #print(k.shape, v.shape)

            r_ind = torch.arange(N).view(1,-1).repeat(B, 1)
            #if self.training:
            r_ind += N * torch.from_numpy(np.random.choice([ii for ii in range(self.dense_mul)], size=(B, N), p=self.dense_prob))
            #else:
            #    r_ind += (torch.from_numpy(np.array(self.dense_test_ind)) * N).view(1,-1).repeat(B,1+N//len(self.dense_test_ind))[:,:N]
            r_ind += torch.arange(B).view(-1,1) * N * self.dense_mul
            k, v = k[r_ind.view(-1),...], v[r_ind.view(-1),...]
            #print(k.shape, v.shape)

            k = rearrange(k, '(b n) h c -> b h n c', b=B, n=N)
            v = rearrange(v, '(b n) h c -> b h n c', b=B, n=N)
            #print(k.shape, v.shape)
        '''

        '''if prev_k is not None:
            q, k, v = torch.cat([q.unsqueeze(-1), prev_q], dim=-1), torch.cat([k.unsqueeze(-1), prev_k], dim=-1), torch.cat([v.unsqueeze(-1), prev_v], dim=-1) # B h N C//h 3
            q = torch.sum(self.dense_mix_q.view(1,1,N,1,-1).softmax(dim=-1) * q, dim=-1)
            k = torch.sum(self.dense_mix_k.view(1,1,N,1,-1).softmax(dim=-1) * k, dim=-1)
            v = torch.sum(self.dense_mix_v.view(1,1,N,1,-1).softmax(dim=-1) * v, dim=-1)'''

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) #C)
        x = self.act(x)
        x = self.proj(x)
        #x = self.act(x)
        #x = self.proj2(x)
        x = self.proj_drop(x)
        return x#, q0, k0, v0



class ConvAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., depth=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        #ViT-Ti
        self.h = self.w = 8
        self.c = 3 # SAME AS NUM OF HEADS
        self.ks = 3
        #self.N = 14*14

        #ViT-S
        '''self.h = self.w = 8
        self.c = 6
        self.ks = 3'''

        #ViT-B/32
        '''self.h = self.w = 8
        self.c = 12
        self.ks = 3'''

        self.qkv_linear = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.qkv = nn.Conv2d(self.c, self.c*3, kernel_size=self.ks, groups=1, padding=self.ks//2, bias=qkv_bias)
        #self.qkv1 = nn.Conv2d(self.c, 16, kernel_size=3, groups=1, padding=1, bias=qkv_bias)
        #self.qkv2 = nn.Conv2d(16, self.c*3, kernel_size=1, groups=1, padding=0, bias=qkv_bias)
        #self.act = nn.GELU()
        #self.norm = nn.LayerNorm(dim)

        self.proj = nn.Conv2d(self.c, self.c, kernel_size=self.ks, groups=1, padding=self.ks//2, bias=qkv_bias)
        #self.proj1 = nn.Conv2d(self.c, 16, kernel_size=3, groups=1, padding=1, bias=qkv_bias)
        #self.proj2 = nn.Conv2d(16, self.c, kernel_size=1, groups=1, padding=0, bias=qkv_bias)
        self.proj_linear = nn.Linear(dim, dim)


    def forward(self, x):#, prev_q, prev_k, prev_v):
        B, N, C = x.shape

        qkv_l = self.qkv_linear(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        x = rearrange(x, 'b n (c h w) -> (b n) c h w', c=self.c, h=self.h, w=self.w)#.contiguous()
        qkv = self.qkv(x)
        #qkv = self.qkv2(self.act(self.qkv1(x)))
        qkv = rearrange(qkv, '(b n) (cx c) h w -> cx b c n (h w)', b=B, n=N, cx=3, c=self.num_heads)#.contiguous() # 3 B h N C//h
        '''qkv = rearrange(qkv, '(b n) (cx c) h w -> (b n) cx (c h w)', b=B, n=N, cx=3, c=self.c)
        qkv = self.norm(qkv)
        qkv = rearrange(qkv, '(b n) cx (c h w) -> cx b c n (h w)', b=B, n=N, c=self.num_heads, h=self.h, w=self.w)#.contiguous() # 3 B h N C//h'''

        qkv = (qkv + qkv_l)*0.5
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B h N C//h

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v) # B h N C//h

        x = rearrange(x_attn, 'b c n (h w) -> (b n) c h w', h=self.h, w=self.w)#.contiguous()
        x = self.proj(x)
        #x = self.proj2(self.act(self.proj1(x)))
        x = rearrange(x, '(b n) c h w -> b n (c h w)', b=B, n=N, c=self.num_heads)#.contiguous()
        #x = self.proj_drop(x)

        x_l = x_attn.transpose(1, 2).reshape(B, N, C)
        x_l = self.proj_linear(x_l)

        x = (x + x_l)*0.5
        x = self.proj_drop(x)

        return x#, q0, k0, v0


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0):
        super().__init__()
        self.norm1 = norm_layer(dim)

        seq = 196; ks = 5; self.H = self.W = 14
        #seq = 49; ks = 3; self.H = self.W = 7

        self.attn = Attention(dim, dim_ratio=1, act=False, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, depth=depth)
        #self.attn = ConvAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, depth=depth)
        #self.mlp_tokens = Mlp(197, dim//2, act_layer=act_layer, drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        '''self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)'''
        mlp_hidden_dim = dim #int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        #self.mlp = ConvMlpGeneral(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop, spatial_dim='2d',
        #                            kernel_size=ks, groups=mlp_hidden_dim, other_dim=seq)
        #self.attn2 = Attention(dim, dim_ratio=3, act=True, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, depth=depth)
        #self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        #self.norm2 = norm_layer(seq)
        ##self.attn2 = Attention2(seq_len=seq, dim=dim, num_heads=4, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = Attention(seq, dim_ratio=1, act=False, num_heads=4, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = norm_layer(dim)
        self.mlp3 = Mlp(in_features=dim, hidden_features=2*dim, act_layer=act_layer, drop=drop)
        #self.H = self.W = 14
        self.depth = depth


    def forward(self, x):#, prev_q, prev_k, prev_v):
        #x = x + self.drop_path(self.attn(self.norm1(x))) # B N C
        #x_, prev_q, prev_k, prev_v = self.attn(self.norm1(x), prev_q, prev_k, prev_v)
        ##x__ = self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        ##x = x + self.drop_path((x_ + x__)/2.)
        #x = x + self.drop_path(x_)

        x = x + self.drop_path(self.attn(self.norm1(x))) # B N C
        #x = x + self.drop_path(self.mlp(self.norm3(x)))

        #x = x + self.drop_path(self.attn2(self.norm2(x)))
        
        #x = x + self.drop_path(self.mlp2(self.norm4(x)))
        #x = x + self.drop_path(self.attn2(self.norm2(x.transpose(-1,-2))).transpose(-1,-2))
        x = x + self.drop_path(self.attn2(self.norm2(x).transpose(-1,-2)).transpose(-1,-2))

        x = x + self.drop_path(self.mlp3(self.norm3(x)))

        '''res = x
        x = self.norm2(x) # B N C
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        x = self.mlp(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = res + self.drop_path(x)'''


        #x = x + self.drop_path(self.attn2(self.norm2(x).transpose(-1,-2).contiguous()).transpose(-1,-2).contiguous())
        #x_skip = x
        #x = self.norm2(x)
        #x =torch.cat([self.mlp(x[:,0:1,:]), self.attn2(x[:,1:,:].transpose(-1,-2).contiguous()).transpose(-1,-2).contiguous()], dim=1)
        #x = x_skip + self.drop_path(x)

        #x = x + self.drop_path(self.attn2(self.norm2(x)))

        return x #, prev_q, prev_k, prev_v

########################################################################################################################

'''
class Attention2(nn.Module):
    def __init__(self, seq_len, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = (seq_len - 1) // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(seq_len, seq_len * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.cls_proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        #B, N, C = x.shape
        #x = torch.cat([x, torch. (B,N,1).cuda()],dim=-1)
        cls, x = x[:,:1,:], x[:,1:,:] # B 1 C, B N-1 C
        x = x.transpose(1,2).contiguous() # B C N-1
        B, N, C = x.shape
        #print(x.shape)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B h N C//h

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        cls = self.attn_drop(self.act(self.cls_proj(cls)))
        x = torch.cat([cls, x.transpose(1,2).contiguous()], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        #x = x[...,:-1]
        return x
'''

########################################################################################################################

class Attention3(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., depth=0, dc=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        '''DEPTH_CONNECTIONS = dc #3
        self.dense_mul = min(depth+1,DEPTH_CONNECTIONS)

        self.dense_mix_q = nn.Parameter(torch.FloatTensor(197, self.dense_mul)) if depth>0 else None
        self.dense_mix_kv = nn.Parameter(torch.FloatTensor(197, self.dense_mul)) if depth>0 else None'''


    def forward(self, xq, xkv): #x, prev_x):
        B, N, C = xq.shape

        '''B, N, C = x.shape
        x0 = xq = xkv = x

        if prev_x is not None:
            x = torch.cat([x.unsqueeze(-1), prev_x], dim=-1) # B N C 3
            xq = torch.sum(self.dense_mix_q.view(1,N,1,-1).softmax(dim=-1) * x, dim=-1)
            xkv = torch.sum(self.dense_mix_kv.view(1,N,1,-1).softmax(dim=-1) * x, dim=-1)'''

        q = self.q(xq).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(xkv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x#, x0


class Block3(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0, dc=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention3(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                depth=depth, dc=dc)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.depth = depth

        DEPTH_CONNECTIONS = dc #3
        self.dense_mul = min(depth+1,DEPTH_CONNECTIONS)
        self.dense_mix_q = nn.Parameter(torch.FloatTensor(197, self.dense_mul)) if depth>0 else None
        self.dense_mix_kv = nn.Parameter(torch.FloatTensor(197, self.dense_mul)) if depth>0 else None


    def forward(self, x, prev_x):

        x0 = xq = xkv = x
        B, N, C = x.shape
        if prev_x is not None:
            x_ = torch.cat([x.unsqueeze(-1), prev_x], dim=-1) # B N C 3
            xq = torch.sum(self.dense_mix_q.view(1,N,1,-1).softmax(dim=-1) * x_, dim=-1)
            xkv = torch.sum(self.dense_mix_kv.view(1,N,1,-1).softmax(dim=-1) * x_, dim=-1)
        x_ = self.attn(self.norm1(xq), self.norm1(xkv))

        #x_, prev_x = self.attn(self.norm1(x), prev_x)
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, x0 #, prev_x


########################################################################################################################


class Block4(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0, dim_map=None):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if dim_map:
            self.cls_map_proj = nn.Linear(dim_map, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.depth = depth


    def forward(self, x):
        x_ = self.norm1(x)
        B, N, C = x_.shape
        qkv = self.qkv(x_).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_ = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_ = self.proj(x_)
        x_ = self.proj_drop(x_)
        x = x + self.drop_path(x_)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def before_qkv(self, x):
        x_ = self.norm1(x)
        B, N, C = x_.shape
        qkv = self.qkv(x_).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def after_qkv(self, x, q, k, v):
        B, N, C = x.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_ = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_ = self.proj(x_)
        x_ = self.proj_drop(x_)
        x = x + self.drop_path(x_)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossVisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=(16,32), in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()

        self.depth = depth

        self.vit_16 = VisionTransformer(patch_size=patch_size[0], num_classes=0, embed_dim=embed_dim[0], embed_dim_map=embed_dim[1],
                        depth=depth,num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=PatchEmbed,
                        norm_layer=norm_layer, act_layer=act_layer, weight_init=weight_init)
        self.vit_32 = VisionTransformer(patch_size=patch_size[1], num_classes=0, embed_dim=embed_dim[1], embed_dim_map=embed_dim[0],
                        depth=depth,num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=PatchEmbed,
                        norm_layer=norm_layer, act_layer=act_layer, weight_init=weight_init)
        #self.head = nn.Linear(2 * embed_dim, num_classes)
        self.head = nn.Linear(embed_dim[0]+embed_dim[1], num_classes)

    def forward(self, x):
        x_16 = self.vit_16.forward_start(x) # B N C
        x_32 = self.vit_32.forward_start(x)

        for d in range(self.depth):
            x_16 = torch.cat([x_16, self.vit_16.blocks[d].cls_map_proj(x_32[:,:1,:].detach())], dim=1)
            x_32 = torch.cat([x_32, self.vit_32.blocks[d].cls_map_proj(x_16[:,:1,:].detach())], dim=1)

            q_16, k_16, v_16 = self.vit_16.blocks[d].before_qkv(x_16)
            q_32, k_32, v_32 = self.vit_32.blocks[d].before_qkv(x_32)

            '''if d in [0,self.depth-1] or d%2==0:
                x_16 = self.vit_16.blocks[d].after_qkv(x_16, q_16, k_16, v_16)
                x_32 = self.vit_32.blocks[d].after_qkv(x_32, q_32, k_32, v_32)
            else:
                x_16 = self.vit_16.blocks[d].after_qkv(x_16, q_16, k_32, v_32)
                x_32 = self.vit_32.blocks[d].after_qkv(x_32, q_32, k_16, v_16)'''

            x_16 = self.vit_16.blocks[d].after_qkv(x_16, q_16, k_16, v_16)
            x_32 = self.vit_32.blocks[d].after_qkv(x_32, q_32, k_32, v_32)

            x_16 = x_16[:,:-1,:]
            x_32 = x_32[:,:-1,:]

        x_16 = self.vit_16.forward_end(x_16)
        x_32 = self.vit_32.forward_end(x_32)

        #x = (x_16 + x_32)/2
        x = self.head(torch.cat([x_16, x_32], dim=-1))
        return x


########################################################################################################################


class Block5(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0, dim_map=None):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.depth = depth

        self.H = self.W = 7 #14
        self.sh = self.sw = 2 #2
        self.N = self.H * self.W
        self.M = self.sh * self.sw

        self.mix = nn.Parameter(torch.FloatTensor(1, self.N, self.M, 1, 2))


    def forward(self, x):
        x_ = self.norm1(x)
        #B, N, C = x_.shape
        #x_ = rearrange(x_, 'b (h sh w sw) c -> b (h w) (sh sw) c', h=self.H, w=self.W, sh=self.sh, sw=self.sw) # B N M C
        B, N, M, C = x_.shape

        qkv = self.qkv(x_)#.reshape(B, N, M, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        qkv_m = rearrange(qkv, 'b n m (k h c) -> k (b n) h m c', k=3, h=self.num_heads, c=C // self.num_heads)
        qkv_n = rearrange(qkv, 'b n m (k h c) -> k (b m) h n c', k=3, h=self.num_heads, c=C // self.num_heads)
        qm, km, vm = qkv_m[0], qkv_m[1], qkv_m[2] # BN h M C
        qn, kn, vn = qkv_n[0], qkv_n[1], qkv_n[2] # BM h N C

        attn_m = (qm @ km.transpose(-2, -1)) * self.scale
        attn_m = attn_m.softmax(dim=-1)
        attn_m = self.attn_drop(attn_m)
        x_m = (attn_m @ vm)#.transpose(1, 2).reshape(B, N, C)
        x_m = rearrange(x_m, '(b n) h m c -> b n m (h c)', n=self.N)

        attn_n = (qn @ kn.transpose(-2, -1)) * self.scale
        attn_n = attn_n.softmax(dim=-1)
        attn_n = self.attn_drop(attn_n)
        x_n = (attn_n @ vn)#.transpose(1, 2).reshape(B, N, C)
        x_n = rearrange(x_n, '(b m) h n c -> b n m (h c)', m=self.M)

        #mix_nm = self.mix.sigmoid()
        #x_ = x_n * mix_nm + x_m * (1.-mix_nm)
        x_ = torch.sum(torch.stack([x_n, x_m], dim=-1) * self.mix.softmax(dim=-1), dim=-1)

        x_ = self.proj(x_)
        x_ = self.proj_drop(x_)
        x = x + self.drop_path(x_)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



########################################################################################################################

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, embed_dim_map=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        #self.DEPTH_CONNECTIONS = 8 #3

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 0 ######### 2 if distilled else 1 #0 #2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = None #######nn.Parameter(torch.zeros(1, 1, embed_dim)) #None #nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = None ########nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None #None #nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        #self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        #self.H = self.W = 7 #14 #7
        #self.sh = self.sw = 2 #2

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, depth=i)
            for i in range(depth)])
        '''self.blocks = nn.Sequential(*[
            Block3(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                depth=i, dc=self.DEPTH_CONNECTIONS)
            for i in range(depth)])'''
        '''self.blocks = nn.Sequential(*[
            Block4(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                depth=i, dim_map=embed_dim_map)
            for i in range(depth)])'''
        '''self.blocks = nn.Sequential(*[
            Block5(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, depth=i)
            for i in range(depth)])'''
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            #trunc_normal_(self.cls_token, std=.02) ###########
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        ################
        '''cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)'''


        x = self.pos_drop(x + self.pos_embed)

        #x = rearrange(x, 'b (h sh w sw) c -> b (h w) (sh sw) c', h=self.H, w=self.W, sh=self.sh, sw=self.sw) # B N M C

        #x = rearrange(x, 'b (h sh w sw) c -> b (h w sh sw) c', h=2, w=2, sh=7, sw=7) # B N C
        #x = rearrange(x, 'b n (c h sh w sw) -> b n (h w c sh sw)', c=3, h=2, w=2, sh=4, sw=4) # B N C

        #x = rearrange(x, 'b n (c hw) -> b n (hw c)', c=3) # B N C

        x = self.blocks(x)

        ######################################################################################
        '''prev_q, prev_k, prev_v = None, None, None # B h N C//h
        for blk in self.blocks:
            x, new_q, new_k, new_v = blk(x, prev_q, prev_k, prev_v)
            if prev_k is None:
                #prev_k, prev_v = new_k, new_v
                prev_q, prev_k, prev_v = new_q.unsqueeze(-1), new_k.unsqueeze(-1), new_v.unsqueeze(-1)
            else:
                #ch = new_k.shape[-1]
                #prev_k = torch.cat([new_k, prev_k], dim=-1)[...,:ch * min(blk.depth+1, 3-1)]
                #prev_v = torch.cat([new_v, prev_v], dim=-1)[...,:ch * min(blk.depth+1, 3-1)]

                #nt = new_k.shape[-2]
                #prev_k = torch.cat([new_k, prev_k], dim=-2)[:,:,:nt * min(blk.depth+1, 3-1),:]
                #prev_v = torch.cat([new_v, prev_v], dim=-2)[:,:,:nt * min(blk.depth+1, 3-1),:]

                prev_q = torch.cat([new_q.unsqueeze(-1), prev_q], dim=-1)[..., :min(blk.depth+1, 3-1)]
                prev_k = torch.cat([new_k.unsqueeze(-1), prev_k], dim=-1)[..., :min(blk.depth+1, 3-1)]
                prev_v = torch.cat([new_v.unsqueeze(-1), prev_v], dim=-1)[..., :min(blk.depth+1, 3-1)]

            #x, _, _ = blk(x, prev_k, prev_v)'''

        '''prev_x= None # B N C
        for blk in self.blocks:
            DEPTH_CONNECTIONS = min(blk.depth+1, self.DEPTH_CONNECTIONS-1)
            x, new_x = blk(x, prev_x)
            if prev_x is None: prev_x = new_x.unsqueeze(-1)
            else: prev_x = torch.cat([new_x.unsqueeze(-1), prev_x], dim=-1)[..., :DEPTH_CONNECTIONS]'''

        ######################################################################################

        x = self.norm(x)
        if self.dist_token is None:
            #return self.pre_logits(x[:, 0])
            return self.pre_logits(x.mean(dim=1)) #########
            #return self.pre_logits(x.mean(dim=(1,2)))
        else:
            return x[:, 0], x[:, 1]


    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x




    def forward_start(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def forward_end(self, x):
        x = self.norm(x)
        if self.dist_token is None:
            x = self.pre_logits(x[:, 0])
        else:
            x = x[:, 0], x[:, 1]

        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


########################################################################################################################


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer,
        #CrossVisionTransformer,
        variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch32_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def cross_vit_tiny_patch16x32_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=(16,32), embed_dim=(192,384), depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_sam_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_sam_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_sam_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_sam_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        'deit_tiny_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        'deit_small_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'deit_base_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
    return model
