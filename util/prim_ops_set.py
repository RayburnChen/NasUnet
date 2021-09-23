from enum import Enum
from typing import Optional

import torch.nn as nn

from util.utils import *

OPS = {
    'none': lambda c_in, c_ot, op_type, dp: ZeroOp(stride=1),
    'identity': lambda c_in, c_ot, op_type, dp: nn.Identity(),
    'avg_pool': lambda c_in, c_ot, op_type, dp: build_ops('avg_pool', op_type),
    'max_pool': lambda c_in, c_ot, op_type, dp: build_ops('max_pool', op_type),

    'cweight': lambda c_in, c_ot, op_type, dp: build_ops('cweight', op_type, c_in, c_ot, dp=dp),
    'dep_conv': lambda c_in, c_ot, op_type, dp: build_ops('dep_conv', op_type, c_in, c_ot, dp=dp),
    'conv': lambda c_in, c_ot, op_type, dp: build_ops('conv', op_type, c_in, c_ot, dp=dp),
    'dil_conv': lambda c_in, c_ot, op_type, dp: build_ops('dil_conv', op_type, c_in, c_ot, dp=dp),
}

DownOps = [
    'avg_pool',
    'max_pool',
    'cweight',
    'dil_conv',
    'dep_conv',
    'conv',
]

UpOps = [
    'cweight',
    'dep_conv',
    'conv',
    'dil_conv',
]

NormOps = [
    'identity',
    'none',
    'cweight',
    'dil_conv',
    'dep_conv',
    'conv',
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
    elif op_name == 'cweight':
        return CWeightOp(c_in, c_ot, stride=stride, use_transpose=use_transpose, output_padding=output_padding,
                         dropout_rate=dp)
    elif op_name == 'dep_conv':
        return ConvOps(c_in, c_ot, stride=stride, use_transpose=use_transpose, output_padding=output_padding,
                       use_depthwise=True, dropout_rate=dp)
    elif op_name == 'conv':
        return ConvOps(c_in, c_ot, stride=stride, use_transpose=use_transpose, output_padding=output_padding,
                       dropout_rate=dp)
    elif op_name == 'dil_conv':
        return ConvOps(c_in, c_ot, stride=stride, use_transpose=use_transpose, output_padding=output_padding,
                       dilation=2, dropout_rate=dp)


class BaseOp(nn.Module):

    def __init__(self, in_channels, out_channels, norm_type='gn', affine=True, act_func='relu', dropout_rate=0, ops_order='weight_norm_act'):
        super(BaseOp, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order
        self.norm_type = norm_type
        self.c = 16
        self.affine = affine

        self.norm = None
        self.activation = None
        self.dropout = None

        # batch norm, group norm, instance norm, layer norm
        if 'norm' in self.ops_list:
            self.build_norm()

        # activation
        if 'act' in self.ops_list:
            self.build_act()

        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=False)

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def norm_before_weight(self):
        for op in self.ops_list:
            if op == 'norm':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def build_norm(self):
        # Ref: <Group Normalization> https://arxiv.org/abs/1803.08494
        # 16 channels for one group is best
        if self.norm_before_weight:
            group = 1 if (self.in_channels % self.c > 0) else 0
            group += self.in_channels // self.c
            if self.norm_type == 'gn':
                self.norm = nn.GroupNorm(group, self.in_channels, affine=self.affine)
            else:
                self.norm = nn.BatchNorm2d(self.in_channels, affine=self.affine)
        else:
            group = 1 if (self.out_channels % self.c > 0) else 0
            group += self.out_channels // self.c
            if self.norm_type == 'gn':
                self.norm = nn.GroupNorm(group, self.out_channels, affine=self.affine)
            else:
                self.norm = nn.BatchNorm2d(self.out_channels, affine=self.affine)

    def build_act(self):
        if self.act_func == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.act_func == 'relu6':
            self.activation = nn.ReLU6(inplace=True)

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == 'norm':
                if self.norm is not None:
                    x = self.norm(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x


class ConvOps(BaseOp):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=False, use_transpose=False, output_padding=0, use_depthwise=False,
                 norm_type='gn', affine=True, act_func='relu', dropout_rate=0,
                 ops_order='weight_norm_act'):
        super(ConvOps, self).__init__(in_channels, out_channels, norm_type, affine, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.use_transpose = use_transpose
        self.use_depthwise = use_depthwise
        self.output_padding = output_padding

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        # 'kernel_size', 'stride', 'padding', 'dilation' can either be 'int' or 'tuple' of int
        if use_transpose:
            if use_depthwise:  # 1. transpose depth-wise conv
                self.depth_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                                     stride=self.stride, padding=padding,
                                                     output_padding=self.output_padding, groups=in_channels,
                                                     bias=self.bias)  # output_padding 1
                self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=self.groups, bias=False)
            else:  # 2. transpose conv
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                               stride=self.stride, padding=padding,
                                               output_padding=self.output_padding, dilation=self.dilation,
                                               bias=self.bias)  # padding 3 output_padding 1
        else:
            if use_depthwise:  # 3. depth-wise conv
                self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size,
                                            stride=self.stride, padding=padding,
                                            dilation=self.dilation, groups=in_channels, bias=False)
                self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                            groups=self.groups, bias=False)
            else:  # 4. conv
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                      stride=self.stride, padding=padding,
                                      dilation=self.dilation, bias=False)

    def weight_call(self, x):
        if self.use_depthwise:
            x = self.depth_conv(x)
            x = self.point_conv(x)
        else:
            x = self.conv(x)
        return x


class CWeightOp(BaseOp):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=None,
                 bias=False, use_transpose=False, output_padding=0, norm_type='gn', affine=True, act_func='relu', dropout_rate=0, ops_order='weight_act'):
        super(CWeightOp, self).__init__(in_channels, out_channels, norm_type, affine, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.use_transpose = use_transpose
        self.output_padding = output_padding

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        # `kernel_size`, `stride`, `padding`, `dilation`
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1 if in_channels < 16 else in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(1 if in_channels < 16 else in_channels // 16, out_channels),
            nn.Sigmoid()
        )
        if stride >= 2:
            if use_transpose:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                               stride=self.stride, padding=padding, output_padding=self.output_padding,
                                               # output_padding 1
                                               bias=False)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=False)
            group = 1 if (out_channels % 16 > 0) else 0
            group += out_channels // 16
            self.norm = nn.GroupNorm(group, out_channels, affine=affine)

    def weight_call(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        rst = self.norm(self.conv(x * y)) if self.stride >= 2 else x * y
        return rst


class ZeroOp(nn.Module):

    def __init__(self, stride):
        super(ZeroOp, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)
