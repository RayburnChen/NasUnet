from enum import Enum
from typing import Optional

import torch.nn as nn

from util.utils import *

OPS = {
    'none': lambda c_in, c_ot, op_type, dp: ZeroOp(stride=1),
    'identity': lambda c_in, c_ot, op_type, dp: nn.Identity(),
    'avg_pool': lambda c_in, c_ot, op_type, dp: build_ops('avg_pool', op_type),
    'max_pool': lambda c_in, c_ot, op_type, dp: build_ops('max_pool', op_type),

    'conv': lambda c_in, c_ot, op_type, dp: build_ops('conv', op_type, c_in, c_ot, dp=dp),
    'dil_conv_2': lambda c_in, c_ot, op_type, dp: build_ops('dil_conv_2', op_type, c_in, c_ot, dp=dp),
    'dil_conv_3': lambda c_in, c_ot, op_type, dp: build_ops('dil_conv_3', op_type, c_in, c_ot, dp=dp),
    'se_conv': lambda c_in, c_ot, op_type, dp: build_ops('se_conv', op_type, c_in, c_ot, dp=dp),

    'up_sample': lambda c_in, c_ot, op_type, dp: nn.Upsample(scale_factor=2, mode='bilinear'),
}

DownOps = [
    # 'avg_pool',
    'max_pool',
    'conv',
    'dil_conv_2',
    'dil_conv_3',
    'se_conv',
]

UpOps = [
    'up_sample',
    'conv',
    'dil_conv_2',
    'dil_conv_3',
    'se_conv',
]

NormOps = [
    # 'avg_pool',
    'max_pool',
    'identity',
    'none',
    'conv',
    'dil_conv_2',
    'dil_conv_3',
    'se_conv',
]


class OpType(Enum):
    UP = UpOps
    DOWN = DownOps
    NORM = NormOps


def build_ops(op_name, op_type: OpType, c_in: Optional[int] = None, c_ot: Optional[int] = None, dp=0):
    stride = 1 if op_type == OpType.NORM else 2
    use_transpose = True if op_type == OpType.UP else False
    output_padding = 1 if op_type == OpType.UP else 0
    if op_name == 'avg_pool':
        return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    elif op_name == 'max_pool':
        return nn.MaxPool2d(3, stride=stride, padding=1)
    elif op_name == 'conv':
        return ConvGnReLU(c_in, c_ot, stride=stride, transpose=use_transpose, output_padding=output_padding, dropout=dp)
    elif op_name == 'se_conv':
        return ConvGnSeReLU(c_in, c_ot, stride=stride, transpose=use_transpose, output_padding=output_padding, dropout=dp)
    elif op_name == 'dil_conv_2':
        return ConvGnReLU(c_in, c_ot, stride=stride, transpose=use_transpose, output_padding=output_padding, dilation=2, dropout=dp)
    elif op_name == 'dil_conv_3':
        return ConvGnReLU(c_in, c_ot, stride=stride, transpose=use_transpose, output_padding=output_padding, dilation=3, dropout=dp)
    else:
        raise NotImplementedError()


class Conv(nn.Sequential):

    def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, transpose=False, output_padding=0, dropout=0):
        conv = build_weight(c_in, c_ot, kernel_size, stride, dilation, transpose, output_padding, dropout)
        super().__init__(*conv)


class ConvGnReLU(nn.Sequential):

    def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, transpose=False, output_padding=0,
                 affine=True, dropout=0):
        conv = build_weight(c_in, c_ot, kernel_size, stride, dilation, transpose, output_padding, dropout)
        norm = build_norm(c_ot, affine)
        act = build_activation()
        super().__init__(*conv, norm, act)


class ConvGnSeReLU(nn.Sequential):
    def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, transpose=False, output_padding=0,
                 affine=True, dropout=0):
        conv = build_weight(c_in, c_ot, kernel_size, stride, dilation, transpose, output_padding, dropout)
        norm = build_norm(c_ot, affine)
        se = SEBlock(c_ot)
        act = build_activation()
        super().__init__(*conv, norm, se, act)


def build_weight(c_in, c_ot, kernel_size, stride, dilation, use_transpose, output_padding, dropout_rate):
    padding = get_same_padding(kernel_size)
    padding *= dilation
    ops = []
    if dropout_rate > 0:
        ops.append(nn.Dropout2d(dropout_rate, inplace=False))
    if use_transpose:
        ops.append(nn.ConvTranspose2d(c_in, c_ot, kernel_size=kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, bias=False, output_padding=output_padding))
    else:
        ops.append(nn.Conv2d(c_in, c_ot, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                             bias=False))
    return ops


def build_norm(c_ot, affine):
    c_per_g = 16
    group = 1 if (c_ot % c_per_g > 0) else 0
    group += c_ot // c_per_g
    return nn.GroupNorm(group, c_ot, affine=affine)


def build_activation():
    return nn.ReLU(inplace=True)


class ZeroOp(nn.Module):

    def __init__(self, stride):
        super(ZeroOp, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class SEBlock(nn.Module):
    # credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    def __init__(self, c, r=16):
        super().__init__()
        self.mid = c // r if c > r else 1
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, self.mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

# class BaseOp(nn.Module):
#
#     def __init__(self, c_in, c_ot, norm_type='gn', affine=True, act_func='relu', dropout_rate=0,
#                  ops_order='weight_norm_act'):
#         super(BaseOp, self).__init__()
#
#         self.c_in = c_in
#         self.c_ot = c_ot
#
#         self.act_func = act_func
#         self.dropout_rate = dropout_rate
#         self.ops_order = ops_order
#         self.norm_type = norm_type
#         self.c = 16
#         self.affine = affine
#
#         self.weight = None
#         self.norm = None
#         self.activation = None
#         self.dropout = None
#         self.seq = None
#
#     @property
#     def ops_list(self):
#         return self.ops_order.split('_')
#
#     @property
#     def norm_before_weight(self):
#         for op in self.ops_list:
#             if op == 'norm':
#                 return True
#             elif op == 'weight':
#                 return False
#         raise ValueError('Invalid ops_order: %s' % self.ops_order)
#
#     def build_weight(self):
#         raise NotImplementedError()
#
#     def build_norm(self):
#         # Ref: <Group Normalization> https://arxiv.org/abs/1803.08494
#         # 16 channels for one group is best
#         if self.norm_before_weight:
#             group = 1 if (self.c_in % self.c > 0) else 0
#             group += self.c_in // self.c
#             if self.norm_type == 'gn':
#                 self.norm = nn.GroupNorm(group, self.c_in, affine=self.affine)
#             else:
#                 self.norm = nn.BatchNorm2d(self.c_in, affine=self.affine)
#         else:
#             group = 1 if (self.c_ot % self.c > 0) else 0
#             group += self.c_ot // self.c
#             if self.norm_type == 'gn':
#                 self.norm = nn.GroupNorm(group, self.c_ot, affine=self.affine)
#             else:
#                 self.norm = nn.BatchNorm2d(self.c_ot, affine=self.affine)
#
#     def build_act(self):
#         if self.act_func == 'relu':
#             self.activation = nn.ReLU(inplace=True)
#         elif self.act_func == 'relu6':
#             self.activation = nn.ReLU6(inplace=True)
#
#     def build_dropout(self):
#         if self.dropout_rate > 0:
#             self.dropout = nn.Dropout2d(self.dropout_rate, inplace=False)
#
#     def build_seq(self):
#         ops = []
#         for op in self.ops_list:
#             if op == 'weight' and len(self.weight):
#                 # dropout before weight operation
#                 if self.dropout is not None:
#                     ops.append(self.dropout)
#                 ops.extend(self.weight)
#             elif op == 'norm' and self.norm is not None:
#                 ops.append(self.norm)
#             elif op == 'act' and self.activation is not None:
#                 ops.append(self.activation)
#             else:
#                 raise ValueError('Unrecognized op: %s' % op)
#         self.seq = nn.Sequential(*ops)
#
#     def forward(self, x):
#         return self.seq(x)


# class ConvOps(BaseOp):
#
#     def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, groups=1,
#                  use_transpose=False, output_padding=0, norm_type='gn', affine=True, act_func='relu',
#                  dropout_rate=0,
#                  ops_order='weight_norm_act'):
#         super(ConvOps, self).__init__(c_in, c_ot, norm_type, affine, act_func, dropout_rate, ops_order)
#
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.dilation = dilation
#         self.groups = groups
#         self.use_transpose = use_transpose
#         self.output_padding = output_padding
#
#         padding = get_same_padding(self.kernel_size)
#         padding *= self.dilation
#         self.padding = padding
#
#         if use_transpose:
#             self.conv = nn.ConvTranspose2d(c_in, c_ot, kernel_size=self.kernel_size, stride=self.stride,
#                                            padding=padding, dilation=self.dilation, bias=False,
#                                            output_padding=self.output_padding)
#         else:
#             self.conv = nn.Conv2d(c_in, c_ot, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
#                                   dilation=self.dilation, bias=False)
#
#     def build_weight(self):
#         self.weight = [self.conv]
#
#     def build(self):
#         self.build_norm()
#         self.build_act()
#         super().build()


# class SeConvOps(ConvOps):
#
#     def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, groups=1, use_transpose=False, output_padding=0,
#                  norm_type='gn', affine=True, act_func='relu', dropout_rate=0, ops_order='weight_norm_act'):
#         super(SeConvOps, self).__init__(c_in, c_ot, kernel_size, stride, dilation, groups, use_transpose, output_padding, norm_type, affine, act_func, dropout_rate, ops_order)
#         self.weight = [self.conv, SEBlock(c_ot)]
#         super().build()


# class CWeightOp(BaseOp):
#
#     def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, groups=None,
#                  bias=False, use_transpose=False, output_padding=0, norm_type='gn', affine=True, act_func='relu',
#                  dropout_rate=0, ops_order='weight_act'):
#         super(CWeightOp, self).__init__(c_in, c_ot, norm_type, affine, act_func, dropout_rate, ops_order)
#
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias
#         self.use_transpose = use_transpose
#         self.output_padding = output_padding
#
#         padding = get_same_padding(self.kernel_size)
#         if isinstance(padding, int):
#             padding *= self.dilation
#         else:
#             padding[0] *= self.dilation
#             padding[1] *= self.dilation
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(c_in, 1 if c_in < 16 else c_in // 16),
#             nn.ReLU(inplace=True),
#             nn.Linear(1 if c_in < 16 else c_in // 16, c_ot),
#             nn.Sigmoid()
#         )
#         if stride >= 2:
#             if use_transpose:
#                 self.conv = nn.ConvTranspose2d(c_in, c_ot, kernel_size=self.kernel_size, stride=self.stride,
#                                                padding=padding, bias=False, output_padding=self.output_padding)
#             else:
#                 self.conv = nn.Conv2d(c_in, c_ot, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
#             group = 1 if (c_ot % 16 > 0) else 0
#             group += c_ot // 16
#             self.norm = nn.GroupNorm(group, c_ot, affine=affine)
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         rst = self.norm(self.conv(x * y)) if self.stride >= 2 else x * y
#         rst = self.activation(rst)
#         return rst


# class ShuffleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, stride, dropout_rate):
#         super(ShuffleConv, self).__init__()
#         self.dropout_rate = dropout_rate
#         self.stride = stride
#         g = 1 if (out_channels % 16 > 0) else 0
#         g += out_channels // 16
#         self.g = g
#
#         mid_channels = out_channels // 4
#
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=g, bias=False)
#         self.bn1 = nn.BatchNorm2d(mid_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False)
#         self.bn2 = nn.BatchNorm2d(mid_channels)
#         self.relu2 = nn.ReLU(inplace=True)
#
#         self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, groups=g, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu3 = nn.ReLU(inplace=True)
#
#         if stride > 1:
#             self.down_sample = nn.AvgPool2d(3, stride=stride, padding=1)
#
#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = channel_shuffle(out, self.g)
#         out = self.relu2(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         if self.stride > 1:
#             res = self.down_sample(x)
#         else:
#             res = x
#         out = self.relu3(out + res)
#         return out
