from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from util.utils import *

OPS = {
    'none': lambda c_in, c_ot, op_type, dp: ZeroOp(stride=1),
    'identity': lambda c_in, c_ot, op_type, dp: nn.Identity(),
    'avg_pool': lambda c_in, c_ot, op_type, dp: build_ops('avg_pool', op_type),
    'max_pool': lambda c_in, c_ot, op_type, dp: build_ops('max_pool', op_type),
    'up_sample': lambda c_in, c_ot, op_type, dp: nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

    'conv': lambda c_in, c_ot, op_type, dp: build_ops('conv', op_type, c_in, c_ot, dp=dp),
    'dil_conv_2': lambda c_in, c_ot, op_type, dp: build_ops('dil_conv_2', op_type, c_in, c_ot, dp=dp),
    'dil_conv_3': lambda c_in, c_ot, op_type, dp: build_ops('dil_conv_3', op_type, c_in, c_ot, dp=dp),
    'se_conv': lambda c_in, c_ot, op_type, dp: build_ops('se_conv', op_type, c_in, c_ot, dp=dp),
    'dep_sep_conv': lambda c_in, c_ot, op_type, dp: build_ops('dep_sep_conv', op_type, c_in, c_ot, dp=dp),
}

DownOps = [
    'conv',
    'dil_conv_2',
    'dil_conv_3',
    'se_conv',
    'dep_sep_conv',
]

UpOps = [
    'conv',
    'dil_conv_2',
    'dil_conv_3',
    'se_conv',
    'dep_sep_conv',
]

NormOps = [
    'avg_pool',
    'identity',
    'none',
    'conv',
    'dil_conv_2',
    'dil_conv_3',
    'se_conv',
    'dep_sep_conv',
]


class OpType(Enum):
    UP = {'id': 1, 'ops': UpOps}
    DOWN = {'id': 2, 'ops': DownOps}
    NORM = {'id': 3, 'ops': NormOps}


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
    elif op_name == 'dep_sep_conv':
        return DepSepConv(c_in, c_ot, stride=stride, transpose=use_transpose, output_padding=output_padding, dropout=dp)
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


class DepSepConv(nn.Sequential):
    def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, transpose=False, output_padding=0,
                 affine=True, dropout=0):
        depth_conv = build_weight(c_in, c_in, kernel_size, stride, dilation, transpose, output_padding, dropout, groups=c_in)
        depth_norm = build_norm(c_in, affine)
        depth_act = build_activation()
        point_conv = build_weight(c_in, c_ot, 1, 1, 1, False, 0, dropout)
        point_norm = build_norm(c_ot, affine)
        point_act = build_activation()
        super().__init__(*depth_conv, depth_norm, depth_act, *point_conv, point_norm, point_act)


def build_weight(c_in, c_ot, kernel_size, stride, dilation, use_transpose, output_padding, dropout_rate, groups=1):
    padding = get_same_padding(kernel_size)
    padding *= dilation
    ops = []
    if dropout_rate > 0:
        ops.append(nn.Dropout2d(dropout_rate, inplace=False))
    if use_transpose:
        ops.append(nn.ConvTranspose2d(c_in, c_ot, kernel_size=kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, bias=False, output_padding=output_padding, groups=groups))
    else:
        ops.append(nn.Conv2d(c_in, c_ot, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                             bias=False, groups=groups))
    return ops


def build_norm(c_ot, affine):
    return nn.BatchNorm2d(c_ot, affine=affine)


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


class ExpandBlock(nn.Module):

    def __init__(self, c_in0, c_in1):
        super().__init__()
        self.c_same = c_in0 == c_in1
        self.pool1 = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        if not self.c_same:
            self.pool2 = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x):
        if self.c_same:
            return self.pool1(x)
        else:
            return torch.cat([self.pool1(x), self.pool2(x)], dim=1)


class ShrinkBlock(nn.Module):

    def __init__(self, c_in, c_ot):
        super().__init__()

        self.conv = nn.Conv2d(c_in, c_ot, kernel_size=3, padding=1, bias=False)
        self.norm = build_norm(c_ot, True)
        self.act = build_activation()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class RectifyBlock(nn.Module):

    def __init__(self, c_in, c_ot, c_res, cell_type='down'):
        super().__init__()
        self.cell_type = cell_type
        self.c_same = c_ot == c_res

        self.conv = nn.Conv2d(c_in, c_ot, kernel_size=3, padding=1, bias=False)
        self.norm = build_norm(c_ot, True)
        self.act = build_activation()

        if not self.c_same and self.cell_type == 'up':
            self.rectify = nn.Conv2d(c_res, c_ot, kernel_size=1, bias=False)
        else:
            self.rectify = nn.Identity()

        if self.cell_type == 'up':
            self.skip_path = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), self.rectify)
        else:
            self.skip_path = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False), self.rectify)

    def forward(self, x, in1):
        out = self.conv(x)
        out = self.norm(out)
        if not self.c_same and self.cell_type == 'down':
            residual = torch.cat([in1, in1], dim=1)
        else:
            residual = in1
        out = self.act(out + self.skip_path(residual))
        return out


class PartialConvGnReLU(nn.Module):

    def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, transpose=False, output_padding=0,
                 affine=True, dropout=0, op_type=OpType.NORM):
        super().__init__()

        k = 4
        g = 1 if (c_ot % 16 > 0) else 0
        g += c_ot // 16
        self.g = g
        c_part = c_ot // k
        self.stride = stride

        self.conv1 = nn.Conv2d(c_in, c_part, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c_part)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_part, c_part, kernel_size=3, stride=stride, padding=1, groups=c_part, bias=False)
        # self.conv2 = build_weight(c_part, c_part, kernel_size, stride, dilation, transpose, output_padding, dropout, groups=c_part)[-1]
        self.bn2 = nn.BatchNorm2d(c_part)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(c_part, c_ot, kernel_size=1, groups=g, bias=False)
        self.bn3 = nn.BatchNorm2d(c_ot)
        self.relu3 = nn.ReLU(inplace=True)

        if self.stride > 1:
            if op_type == OpType.DOWN:
                self.skip_path = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
            else:
                self.skip_path = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, self.g)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride > 1:
            res = self.skip_path(x)
        else:
            res = x
        out = self.relu3(out + res)
        return out
