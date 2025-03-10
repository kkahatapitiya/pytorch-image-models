""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn
from einops import rearrange


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 gate_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class ConvMlpGeneral(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, spatial_dim='2d', kernel_size=3, groups=1,
            act_layer=nn.ReLU, norm_layer=None, drop=0., other_dim=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        #norm_layer=nn.BatchNorm2d

        #Conv = nn.Conv2d if spatial_dim=='2d' else nn.Conv1d
        #self.fc1 = Conv(in_features, hidden_features, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, bias=True)
        #self.fc2 = Conv(hidden_features, out_features, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, bias=True)

        if spatial_dim=='1d':
            #norm_layer=nn.BatchNorm1d
            #self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1, groups=1, padding=0, bias=True)
            #self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, bias=True)

            #self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=kernel_size, groups=1, padding=kernel_size//2, bias=True)
            #self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=kernel_size, groups=1, padding=kernel_size//2, bias=True)

            #self.fc4 = nn.Conv1d(in_features, in_features, kernel_size=kernel_size, groups=in_features, padding=kernel_size//2, bias=True)
            self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1, groups=1, padding=0, bias=True)
            #self.fc3 = nn.Conv1d(hidden_features, hidden_features, kernel_size=kernel_size, groups=hidden_features, padding=kernel_size//2, bias=True)
            self.fc3 = nn.Linear(other_dim, other_dim)
            self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1, groups=1, padding=0, bias=True)

            '''self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, groups=1, padding=0, bias=True)
            self.fc3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, groups=hidden_features, padding=kernel_size//2, bias=True)
            self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, groups=1, padding=0, bias=True)'''

        else:
            #norm_layer=nn.BatchNorm2d
            #self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, groups=1, padding=0, bias=True)
            #self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, bias=True)
            #self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, bias=True)

            #self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1,kernel_size), groups=groups, padding=(0,kernel_size//2), bias=True)
            #self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(kernel_size,1), groups=groups, padding=(kernel_size//2,0), bias=True)

            #self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, bias=True)
            #self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, bias=True)

            #self.fc4 = nn.Conv2d(in_features, in_features, kernel_size=kernel_size, groups=in_features, padding=(kernel_size//2,kernel_size//2), bias=True)
            self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, groups=1, padding=0, bias=True)
            #self.fc3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, groups=hidden_features, padding=kernel_size//2, bias=True)
            self.fc3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, bias=True)
            #self.fc4 = nn.Conv2d(out_features, out_features, kernel_size=kernel_size, groups=out_features, padding=kernel_size//2, bias=True)
            #self.fc3 = nn.Linear(other_dim, other_dim)
            #self.fc3 = nn.Linear(14, 14)
            #self.fc4 = nn.Linear(14, 14)
            self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, groups=1, padding=0, bias=True)


        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.spatial_dim = spatial_dim

    def forward(self, x):
        #if self.spatial_dim=='2d':
        #    b,c,h,w = x.shape
        #    x = rearrange(x, 'b c h w-> (b h) c w')
        '''if self.spatial_dim=='2d' or self.spatial_dim=='1d':
            x = self.fc4(x)
            x = self.norm(x)
            x = self.act(x)'''
        x = self.fc1(x)
        x = self.norm(x)
        ##if self.spatial_dim=='2d':
        ##    x = rearrange(x, '(b h) c w-> b c h w', b=b, h=h)
        ##    x = self.norm(x)
        ##    x = rearrange(x, 'b c h w-> (b w) c h')
        ##else:
        ##    x = self.norm(x)
        x = self.act(x)

        #if self.spatial_dim=='2d' or self.spatial_dim=='1d':
        '''if self.spatial_dim=='2d':
            b,c,h,w = x.shape
            x = x.view(b,c,-1)'''
        #if self.spatial_dim=='1d':
        x = self.fc3(x)
        x = self.norm(x)
        x = self.act(x)
        '''else:
            x = self.fc3(x.transpose(-1,-2)).transpose(-1,-2)
            x = self.fc4(x)
            x = self.norm(x)
            x = self.act(x)'''
        '''if self.spatial_dim=='2d':
            x = x.view(b,c,h,w)'''

        x = self.drop(x)

        #if self.spatial_dim=='2d':
        #    x = rearrange(x, '(b h) c w-> (b w) c h', b=b, h=h)
        x = self.fc2(x)
        #if self.spatial_dim=='2d':
        #    x = rearrange(x, '(b w) c h-> b c h w', b=b, w=w)

        '''x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc4(x)'''

        return x


class ConvMlpGeneralv2(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, spatial_dim='2d', kernel_size=3,
            act_layer=nn.ReLU, norm_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if spatial_dim=='2d':
            self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1,kernel_size), padding=(0,kernel_size//2), bias=True)
            self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(kernel_size,1), padding=(kernel_size//2,0), bias=True)
        else:
            self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
            self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1, padding=0, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
