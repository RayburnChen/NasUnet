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

    'up_sample': lambda c_in, c_ot, op_type, dp: nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
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
        return PartialConvGnReLU(c_in, c_ot, stride=stride, transpose=use_transpose, output_padding=output_padding, dropout=dp, op_type=op_type)
    elif op_name == 'se_conv':
        return PartialConvGnSeReLU(c_in, c_ot, stride=stride, transpose=use_transpose, output_padding=output_padding, dropout=dp, op_type=op_type)
    elif op_name == 'dil_conv_2':
        return PartialConvGnReLU(c_in, c_ot, stride=stride, transpose=use_transpose, output_padding=output_padding, dilation=2, dropout=dp, op_type=op_type)
    elif op_name == 'dil_conv_3':
        return PartialConvGnReLU(c_in, c_ot, stride=stride, transpose=use_transpose, output_padding=output_padding, dilation=3, dropout=dp, op_type=op_type)
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

        # self.conv2 = nn.Conv2d(c_part, c_part, kernel_size=3, stride=stride, padding=1, groups=c_part, bias=False)
        self.conv2 = build_weight(c_part, c_part, kernel_size, stride, dilation, transpose, output_padding, dropout, groups=c_part)[-1]
        self.bn2 = nn.BatchNorm2d(c_part)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(c_part, c_ot, kernel_size=1, groups=g, bias=False)
        self.bn3 = nn.BatchNorm2d(c_ot)
        self.relu3 = nn.ReLU(inplace=True)

        if self.stride > 1:
            if op_type == OpType.DOWN:
                self.skip_path = nn.AvgPool2d(3, stride=stride, padding=1)
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


class PartialConvGnSeReLU(nn.Module):

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

        # self.conv2 = nn.Conv2d(c_part, c_part, kernel_size=3, stride=stride, padding=1, groups=c_part, bias=False)
        self.conv2 = build_weight(c_part, c_part, kernel_size, stride, dilation, transpose, output_padding, dropout, groups=c_part)[-1]
        self.bn2 = nn.BatchNorm2d(c_part)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(c_part, c_ot, kernel_size=1, groups=g, bias=False)
        self.bn3 = nn.BatchNorm2d(c_ot)
        self.relu3 = nn.ReLU(inplace=True)

        self.se = SEBlock(c_ot)

        if self.stride > 1:
            if op_type == OpType.DOWN:
                self.skip_path = nn.AvgPool2d(3, stride=stride, padding=1)
            else:
                self.skip_path = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, self.g)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.stride > 1:
            res = self.skip_path(x)
        else:
            res = x
        out = self.relu3(out + res)
        return out
