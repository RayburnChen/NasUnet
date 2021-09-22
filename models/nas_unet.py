from util.gpu_memory_log import gpu_memory_log
from util.prim_ops_set import *
from .base import BaseNet
from .resnet import BasicBlock


class BuildCell(nn.Module):
    """Build a cell from genotype"""

    def __init__(self, genotype, c_in0, c_in1, c, cell_type, dropout_prob=0):
        super(BuildCell, self).__init__()

        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            # self.preprocess0 = ConvOps(c_in0, c, kernel_size=1, stride=2, ops_order='weight_norm')
            self.preprocess0 = PoolingOp(c_in0, c, kernel_size=3, padding=1, pool_type='max')  # suppose c_in0 == c
        else:
            self.preprocess0 = ConvOps(c_in0, c, kernel_size=3, ops_order='weight_norm_act')
        # self.preprocess1 = ConvOps(c_in1, c, kernel_size=1, ops_order='weight_norm_act')
        self.preprocess1 = nn.Identity()  # suppose c_in1 == c

        if cell_type == 'up':
            op_names, idx = zip(*genotype.up)
            concat = genotype.up_concat
        else:
            op_names, idx = zip(*genotype.down)
            concat = genotype.down_concat

        self.post_process = ConvOps(c * len(concat), c, kernel_size=3, ops_order='weight_norm_act')
        self.dropout_prob = dropout_prob
        self._compile(c, op_names, idx, concat)

    def _compile(self, c, op_names, idx, concat):
        assert len(op_names) == len(idx)
        self._num_meta_node = len(op_names) // 2
        self._concat = concat
        self._multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, idx):
            op = OPS[name](c, None, affine=True, dp=self.dropout_prob)
            self._ops += [op]
        self._indices = idx

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._num_meta_node):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]

            h1 = self._ops[2 * i](h1)
            h2 = self._ops[2 * i + 1](h2)

            s = h1 + h2
            states += [s]

        return self.post_process(torch.cat([states[i] for i in self._concat], dim=1))


class Head(nn.Module):

    def __init__(self, genotype, c_in0, c_in1, nclass):
        super(Head, self).__init__()
        self.up_cell = BuildCell(genotype, c_in0, c_in1, c_in1, cell_type='up')
        self.segmentation_head = ConvOps(c_in1, nclass, kernel_size=3, ops_order='weight')

    def forward(self, s0, ot):
        return self.segmentation_head(self.up_cell(s0, ot))


class NasUnet(BaseNet):
    """Construct a network"""

    def __init__(self, nclass, in_channels, backbone=None, aux=False,
                 c=48, depth=5, dropout_prob=0,
                 supervision=False, genotype=None, double_down_channel=False):

        super(NasUnet, self).__init__(nclass, aux, backbone, norm_layer=nn.GroupNorm)
        self._depth = depth
        self._double_down_channel = double_down_channel
        self._supervision = supervision
        self._multiplier = len(genotype.down_concat)
        self.gamma = genotype.gamma

        assert self._depth >= 2, 'depth must >= 2'
        double_down = 2 if self._double_down_channel else 1
        c_in0, c_in1, c_curr = c, c, c

        self.blocks = nn.ModuleList()
        self.stem0 = ConvOps(in_channels, c_in0, kernel_size=7, ops_order='weight_norm_act')
        stem1_pool = PoolingOp(c_in0, c_in1, kernel_size=3, padding=1, pool_type='max')
        stem1_block = BasicBlock(c_in0, c_in1, stride=1, dilation=1, previous_dilation=1, norm_layer=nn.BatchNorm2d)
        self.stem1 = nn.Sequential(stem1_pool, stem1_block)

        num_filters = []
        down_f = []
        down_block = nn.ModuleList()
        for i in range(self._depth):
            if i == 0:
                filters = [1, 1, int(c_in0), 'stem1']
                down_cell = self.stem1
                down_f.append(filters)
                down_block += [down_cell]
            else:
                c_curr = int(double_down * c_curr)
                filters = [c_in0, c_in1, c_curr, 'down']
                down_cell = BuildCell(genotype, c_in0, c_in1, c_curr, cell_type='down', dropout_prob=dropout_prob)
                down_f.append(filters)
                down_block += [down_cell]
                c_in0, c_in1 = c_in1, c_curr  # down_cell._multiplier

        num_filters.append(down_f)
        self.blocks += [down_block]

        for i in range(1, self._depth):
            up_f = []
            up_block = nn.ModuleList()
            for j in range(self._depth - i):
                gamma_ides = sum(range(self._depth - 2, self._depth - i - 1, -1)) + j
                if i + j < self._depth - 1 and self.gamma[gamma_ides] == 0:
                    filters = [0, 0, 0, 'None']
                    up_cell = None
                else:
                    _, _, head_curr, _ = num_filters[0][j]
                    _, _, head_down, _ = num_filters[i - 1][j + 1]
                    head_in0 = sum([num_filters[k][j][2] for k in range(i)])  # up_cell._multiplier
                    head_in1 = head_down  # up_cell._multiplier
                    filters = [head_in0, head_in1, head_curr, 'up']
                    up_cell = BuildCell(genotype, head_in0, head_in1, head_curr, cell_type='up',
                                        dropout_prob=dropout_prob)
                up_f.append(filters)
                up_block += [up_cell]
            num_filters.append(up_f)
            self.blocks += [up_block]

        self.head_block = nn.ModuleList()

        c_in0 = c
        c_in1 = num_filters[-1][0][2]
        self.head_block += [Head(genotype, c_in0, c_in1, nclass)]

    def forward(self, x):
        cell_out = []
        final_out = []
        for i, block in enumerate(self.blocks):
            for j, cell in enumerate(block):
                if i == 0 and j == 0:
                    # stem0: 1x256x256 -> 32x256x256
                    s0 = self.stem0(x)
                    # stem1: 32x256x256 -> 32x128x128
                    ot = cell(s0)
                elif i == 0 and j == 1:
                    ot = cell(s0, cell_out[-1])
                elif i == 0:
                    ot = cell(cell_out[-2], cell_out[-1])
                else:
                    if i + j < self._depth - 1 and self.gamma[sum(range(self._depth - 2, self._depth - i - 1, -1)) + j] == 0:
                        ot = None
                    else:
                        # ides = [sum(range(self._depth, self._depth - i+1)) + j]
                        ides = [sum(range(self._depth, self._depth - k, -1)) + j for k in range(i)]
                        in0 = torch.cat([cell_out[idx] for idx in ides if cell_out[idx] is not None], dim=1)
                        in1 = cell_out[ides[-1] + 1]
                        ot = cell(in0, in1)
                        if j == 0 and self._supervision:
                            final_out.append(self.head_block[-1](s0, ot))
                cell_out.append(ot)

        del cell_out
        if self._supervision:
            return final_out
        else:
            return [self.head_block[-1](s0, ot)]
