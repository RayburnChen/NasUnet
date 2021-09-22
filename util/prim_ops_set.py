import torch.nn as nn
from util.utils import *

OPS = {
    'none': lambda c_in, c_ot, stride, dp: ZeroOp(stride=stride),
    'identity': lambda c_in, c_ot, stride, dp: nn.Identity(),
    'avg_pool': lambda c_in, c_ot, stride, dp: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool': lambda c_in, c_ot, stride, dp: nn.MaxPool2d(3, stride=stride, padding=1),

    'cweight': lambda c_in, c_ot, stride, dp: CWeightOp(c_in, c_ot, dropout_rate=dp),
    'dep_conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, use_depthwise=True, dropout_rate=dp),
    'conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, has_shuffle=False),
    'shuffle_conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, has_shuffle=True),
    'dil_conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, dilation=2, dropout_rate=dp),

    'down_cweight': lambda c_in, c_ot, stride, dp: CWeightOp(c_in, c_ot, stride=2, dropout_rate=dp),
    'down_dep_conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, stride=2, use_depthwise=True, dropout_rate=dp),
    'down_conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, stride=2, dropout_rate=dp),
    'down_dil_conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, stride=2, dilation=2, dropout_rate=dp),

    'up_cweight': lambda c_in, c_ot, stride, dp: CWeightOp(c_in, c_ot, stride=2, use_transpose=True, output_padding=1, dropout_rate=dp),
    'up_dep_conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, stride=2, use_transpose=True, output_padding=1, use_depthwise=True, dropout_rate=dp),
    'up_conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, stride=2, use_transpose=True, output_padding=1, dropout_rate=dp),
    'up_dil_conv': lambda c_in, c_ot, stride, dp: ConvOps(c_in, c_ot, stride=2, use_transpose=True, output_padding=1, dilation=3, dropout_rate=dp),

}


class AbstractOp(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class BaseOp(AbstractOp):

    def __init__(self, in_channels, out_channels, norm_type='gn', use_norm=True, affine=True,
                 act_func='relu', dropout_rate=0, ops_order='weight_norm_act'):
        super(BaseOp, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_norm = use_norm
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order
        self.norm_type = norm_type
        self.c = 16

        # batch norm, group norm, instance norm, layer norm
        if self.use_norm:
            # Ref: <Group Normalization> https://arxiv.org/abs/1803.08494
            # 16 channels for one group is best
            if self.norm_before_weight:
                group = 1 if (in_channels % self.c > 0) else 0
                group += in_channels // self.c
                if norm_type == 'gn':
                    self.norm = nn.GroupNorm(group, in_channels, affine=affine)
                else:
                    self.norm = nn.BatchNorm2d(in_channels, affine=affine)
            else:
                group = 1 if (out_channels % self.c > 0) else 0
                group += out_channels // self.c
                if norm_type == 'gn':
                    self.norm = nn.GroupNorm(group, out_channels, affine=affine)
                else:
                    self.norm = nn.BatchNorm2d(out_channels, affine=affine)
        else:
            self.norm = None

        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU(inplace=False)
            else:
                self.activation = nn.ReLU(inplace=True)
        elif act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU6(inplace=False)
            else:
                self.activation = nn.ReLU6(inplace=True)
        else:
            self.activation = None

        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=False)
        else:
            self.dropout = None

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

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_norm': self.use_norm,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    @staticmethod
    def is_zero_ops():
        return False

    def get_flops(self, x):
        raise NotImplementedError

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
                 bias=False, has_shuffle=False, use_transpose=False, output_padding=0, use_depthwise=False,
                 norm_type='gn', use_norm=True, affine=True, act_func='relu', dropout_rate=0,
                 ops_order='weight_norm_act'):
        super(ConvOps, self).__init__(in_channels, out_channels, norm_type, use_norm, affine, act_func, dropout_rate,
                                      ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
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

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        basic_str = 'Conv'
        basic_str = 'Dilation' + basic_str if self.dilation > 1 else basic_str
        basic_str = 'Depth' + basic_str if self.use_depthwise else basic_str
        basic_str = 'Group' + basic_str if self.groups > 1 else basic_str
        basic_str = 'Tran' + basic_str if self.use_transpose else basic_str
        basic_str = '%dx%d_' % (kernel_size[0], kernel_size[1]) + basic_str
        return basic_str

    @property
    def config(self):
        config = {
            'name': ConvOps.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            'depth_wise': self.use_depthwise,
            'transpose': self.use_transpose,
        }
        config.update(super(ConvOps, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return ConvOps(**config)

    def weight_call(self, x):
        if self.use_depthwise:
            x = self.depth_conv(x)
            x = self.point_conv(x)
        else:
            x = self.conv(x)
        if self.has_shuffle and self.groups > 1:
            x = shuffle_layer(x, self.groups)
        return x


class CWeightOp(BaseOp):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=None,
                 bias=False, has_shuffle=False, use_transpose=False, output_padding=0, norm_type='gn',
                 use_norm=False, affine=True, act_func='relu', dropout_rate=0, ops_order='weight_act'):
        super(CWeightOp, self).__init__(in_channels, out_channels, norm_type, use_norm, affine, act_func, dropout_rate,
                                        ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
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

    @property
    def unit_str(self):
        basic_str = 'ChannelWeight'
        basic_str = 'Tran' + basic_str if self.use_transpose else basic_str
        return basic_str

    @staticmethod
    def build_from_config(config):
        return CWeightOp(**config)

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






