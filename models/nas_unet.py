from util.gpu_memory_log import gpu_memory_log
from util.prim_ops_set import *
from .base import BaseNet
from .resnet import BasicBlock


class BuildCell(nn.Module):
    """Build a cell from genotype"""

    def __init__(self, genotype, c_in0, c_in1, c, cell_type, dropout_prob=0):
        super(BuildCell, self).__init__()
        if cell_type == 'up':
            op_names, idx = zip(*genotype.up)
            concat = genotype.up_concat
        else:
            op_names, idx = zip(*genotype.down)
            concat = genotype.down_concat
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
        states = [s0, s1]
        for i in range(self._num_meta_node):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]

            h1 = self._ops[2 * i](h1)
            h2 = self._ops[2 * i + 1](h2)

            s = h1 + h2
            states += [s]

        result = torch.cat([states[i] for i in self._concat], dim=1)
        return result


class Head(nn.Module):

    def __init__(self, c_last, nclass):
        super(Head, self).__init__()
        self.head = ConvOps(c_last, nclass, kernel_size=1, ops_order='weight')

    def forward(self, ot):
        return self.head(ot)


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
        c_s0, c_s1 = self._multiplier * c, self._multiplier * c
        c_in0, c_in1, c_curr = c_s0, c_s1, c

        self.blocks = nn.ModuleList()
        self.blocks_in_0 = nn.ModuleList()
        self.blocks_in_1 = nn.ModuleList()
        self.stem0 = ConvOps(in_channels, c_in0, kernel_size=1, ops_order='weight_norm')
        skip_down = ConvOps(c_in0, c_in1, kernel_size=1, stride=2, ops_order='weight_norm')
        self.stem1 = BasicBlock(c_in0, c_in1, stride=2, dilation=1, downsample=skip_down, previous_dilation=1,
                                norm_layer=nn.BatchNorm2d)

        self.num_filters = []
        down_f = []
        down_block = nn.ModuleList()
        in_0_prep = nn.ModuleList()
        in_1_prep = nn.ModuleList()
        for i in range(self._depth):
            if i == 0:
                filters = [1, 1, int(c_in0 / self._multiplier), 'stem0']
                down_cell = self.stem0
                down_f.append(filters)
                down_block += [down_cell]
                in_0_prep += [None]
                in_1_prep += [None]
            elif i == 1:
                filters = [1, 1, int(c_in1 / self._multiplier), 'stem1']
                down_cell = self.stem1
                down_f.append(filters)
                down_block += [down_cell]
                in_0_prep += [None]
                in_1_prep += [None]
            else:
                c_curr = int(double_down * c_curr)
                prep_0, prep_1, down_cell = build_cell(genotype, c_in0, c_in1, c_curr, cell_type='down', dropout_prob=dropout_prob)
                filters = [c_in0, c_in1, c_curr, 'down']
                down_f.append(filters)
                down_block += [down_cell]
                in_0_prep += [prep_0]
                in_1_prep += [prep_1]
                c_in0, c_in1 = c_in1, self._multiplier * c_curr  # down_cell._multiplier

        self.num_filters.append(down_f)
        self.blocks += [down_block]
        self.blocks_in_0 += [in_0_prep]
        self.blocks_in_1 += [in_1_prep]

        for i in range(1, self._depth):
            up_f = []
            up_block = nn.ModuleList()
            in_0_prep = nn.ModuleList()
            in_1_prep = nn.ModuleList()
            for j in range(self._depth - i):
                gamma_ides = sum(range(self._depth - 2, self._depth - i - 1, -1)) + j
                if i + j < self._depth - 1 and self.gamma[gamma_ides] == 0:
                    filters = [0, 0, 0, 'None']
                    up_cell = None
                    in_0_prep += [None]
                    in_1_prep += [None]
                else:
                    _, _, head_curr, _ = self.num_filters[0][j]
                    _, _, head_down, _ = self.num_filters[i - 1][j + 1]
                    # head_in0 = self._multiplier * sum([num_filters[i-1][j][2]])  # up_cell._multiplier
                    head_in0 = self._multiplier * sum([self.num_filters[k][j][2] for k in range(i)])  # up_cell._multiplier
                    head_in1 = self._multiplier * head_down  # up_cell._multiplier
                    prep_0, prep_1, up_cell = build_cell(genotype, head_in0, head_in1, head_curr, cell_type='up',
                                        dropout_prob=dropout_prob)
                    filters = [head_in0, head_in1, head_curr, 'up']
                    in_0_prep += [prep_0]
                    in_1_prep += [prep_1]
                up_f.append(filters)
                up_block += [up_cell]
            self.num_filters.append(up_f)
            self.blocks += [up_block]
            self.blocks_in_0 += [in_0_prep]
            self.blocks_in_1 += [in_1_prep]

        self.head_block = nn.ModuleList()
        # if self._supervision:
        #     for i in range(1, depth):
        #         c_last = self._multiplier * num_filters[i][0][2]
        #         self.head_block += [Head(c_last, nclass)]
        # else:
        c_last = self._multiplier * self.num_filters[-1][0][2]
        self.head_block += [Head(c_last, nclass)]

    def forward(self, x):
        cell_out = []
        final_out = []
        for i, block in enumerate(self.blocks):
            for j, cell in enumerate(block):
                if i == 0 and j == 0:
                    # stem0: 1x256x256 -> 64x256x256
                    ot = cell(x)
                elif i == 0 and j == 1:
                    # stem1: 64x256x256 -> 64x128x128
                    ot = cell(cell_out[-1])
                elif i == 0:
                    prep_0 = self.blocks_in_0[i][j]
                    prep_1 = self.blocks_in_1[i][j]
                    p_0 = prep_0(cell_out[-2])
                    p_1 = prep_1(cell_out[-1])
                    ot = cell(p_0, p_1)
                else:
                    if i + j < self._depth - 1 and self.gamma[
                        sum(range(self._depth - 2, self._depth - i - 1, -1)) + j] == 0:
                        ot = None
                    else:
                        prep_0 = self.blocks_in_0[i][j]
                        prep_1 = self.blocks_in_1[i][j]
                        # ides = [sum(range(self._depth, self._depth - i+1)) + j]
                        ides = [sum(range(self._depth, self._depth - k, -1)) + j for k in range(i)]
                        in0 = torch.cat([cell_out[idx] for idx in ides if cell_out[idx] is not None], dim=1)
                        in1 = cell_out[ides[-1] + 1]
                        ot = cell(prep_0(in0), prep_1(in1))
                        if j == 0 and self._supervision:
                            final_out.append(self.head_block[-1](ot))
                cell_out.append(ot)

        gpu_memory_log()
        del cell_out
        if self._supervision:
            return final_out
        else:
            return [self.head_block[-1](ot)]


def build_cell(genotype, c_in0, c_in1, c_curr, cell_type, dropout_prob):
    # Note: the s0 size is twice than s1!
    prep_0 = ConvOps(c_in0, c_curr, kernel_size=1, stride=2 if cell_type == 'down' else 1, ops_order='act_weight_norm')
    prep_1 = ConvOps(c_in1, c_curr, kernel_size=1, ops_order='act_weight_norm')
    cell = BuildCell(genotype, c_in0, c_in1, c_curr, cell_type=cell_type, dropout_prob=dropout_prob)
    return prep_0, prep_1, cell
