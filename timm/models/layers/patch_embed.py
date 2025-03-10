""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from einops import rearrange
import math

from .helpers import to_2tuple


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.proj2 = nn.Conv2d(in_chans, 1, kernel_size=1, stride=1)
        #self.proj2 = nn.Conv2d(in_chans, 3, kernel_size=2, stride=2)
        self.proj21 = nn.Conv2d(in_chans, 128, kernel_size=4, stride=2, padding=1)
        self.proj22 = nn.Conv2d(128, 3, kernel_size=1, stride=1)
        self.act = nn.GELU()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm2 = norm_layer(128) if norm_layer else nn.Identity()
        #self.pos_embed2 = nn.Parameter(torch.zeros(1, 128, 8, 8))
        #trunc_normal_(self.pos_embed2, std=.02)

        '''self.proj21 = nn.Conv2d(in_chans, 64, kernel_size=2, stride=2) # B 32 112 112
        self.proj22 = nn.Conv2d(64, 4, kernel_size=1, stride=1) # B 32 112 112
        self.act = nn.GELU() #nn.Identity() #nn.GELU()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm2 = norm_layer(64) if norm_layer else nn.Identity()'''

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        #x = self.proj(x)

        #for Vit
        #x1 = self.proj(x) # B 192 14 14
        #x2 = self.proj2(x)
        x2 = self.norm2(self.act(self.proj21(x)))# + self.pos_embed2.repeat(1,1,14,14))) # B 128 112 112
        x2 = self.proj22(x2)
        #x2 = rearrange(x2, 'b c (n1 p1) (n2 p2) -> b (c p1 p2) n1 n2', n1=14, n2=14, p1=16, p2=16)
        x2 = rearrange(x2, 'b c (n1 p1) (n2 p2) -> b (c p1 p2) n1 n2', n1=14, n2=14, p1=8, p2=8)
        #x = x1 + x2
        x = x2

        #for mixer
        '''#x1 = self.proj(x) # B 256 14 14
        x2 = self.norm2(self.act(self.proj21(x)).transpose(1,-1)).transpose(1,-1)
        x2 = self.proj22(x2)
        x2 = rearrange(x2, 'b c (n1 p1) (n2 p2) -> b (c p1 p2) n1 n2', n1=14, n2=14, p1=8, p2=8)
        #x = x1 + x2
        x = x2'''

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
