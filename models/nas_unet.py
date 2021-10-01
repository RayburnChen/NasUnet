from util.gpu_memory_log import gpu_memory_log
from util.prim_ops_set import *
from .base import BaseNet
from .resnet import BasicBlock


class BuildCell(nn.Module):
    """Build a cell from genotype"""

    def __init__(self, genotype, double_down, c_in0, c_in1, c_out, cell_type, dropout_prob=0):
        super(BuildCell, self).__init__()
        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = build_rectify(c_in0, c_in1, cell_type)
            c_part = c_out // double_down
        else:
            self.preprocess0 = ShrinkBlock(c_in0, c_in1)
            c_part = c_out
        self.preprocess1 = nn.Identity()

        if cell_type == 'up':
            op_names, idx = zip(*genotype.up)
            concat = genotype.up_concat
        else:
            op_names, idx = zip(*genotype.down)
            concat = genotype.down_concat

        self.post_process = RectifyBlock(c_part * len(concat), c_out, c_in1, cell_type=cell_type)
        self.dropout_prob = dropout_prob
        self._compile(c_in1, c_part, cell_type, op_names, idx, concat)

    def _compile(self, c, c_part, cell_type, op_names, idx, concat):
        assert len(op_names) == len(idx)
        self._num_meta_node = len(op_names) // 2
        self._concat = concat
        self._multiplier = len(concat)
        self._input_num = 2

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, idx):
            if index < self._input_num:
                if cell_type == 'down':
                    op = OPS[name](c, c_part, OpType.DOWN, self.dropout_prob)
                elif index > 0:
                    op = OPS[name](c, c_part, OpType.UP, self.dropout_prob)
                else:
                    op = OPS[name](c, c_part, OpType.NORM, self.dropout_prob)
            else:
                op = OPS[name](c_part, c_part, OpType.NORM, self.dropout_prob)
            self._ops += [op]
        self._indices = idx

    def forward(self, in0, in1):
        in0 = self.preprocess0(in0)
        in1 = self.preprocess1(in1)

        states = [in0, in1]
        for i in range(self._num_meta_node):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]

            h1 = self._ops[2 * i](h1)
            h2 = self._ops[2 * i + 1](h2)

            s = h1 + h2
            states += [s]
        out = torch.cat([states[i] for i in self._concat], dim=1)
        return self.post_process(out, in1)


class Head(nn.Module):

    def __init__(self, genotype, double_down, c_in0, c_in1, nclass):
        super(Head, self).__init__()
        self.up_cell = BuildCell(genotype, double_down, c_in0, c_in1, c_in1, cell_type='up')
        self.segmentation_head = Conv(c_in1, nclass, kernel_size=3)

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
        self.stem0 = ConvGnReLU(in_channels, c_in0, kernel_size=7)
        stem1_pool = nn.MaxPool2d(3, stride=2, padding=1)
        stem1_block = BasicBlock(c_in0, c_in1, stride=1, dilation=1, previous_dilation=1, norm_layer=nn.BatchNorm2d)
        self.stem1 = nn.Sequential(stem1_pool, stem1_block)

        num_filters = []
        down_f = []
        down_block = nn.ModuleList()
        for i in range(self._depth):
            if i == 0:
                filters = [1, 1, int(c_in1), 'stem1']
                down_cell = self.stem1
                down_f.append(filters)
                down_block += [down_cell]
            else:
                c_curr = int(double_down * c_curr)
                filters = [c_in0, c_in1, c_curr, 'down']
                down_cell = BuildCell(genotype, double_down, c_in0, c_in1, c_curr, cell_type='down', dropout_prob=dropout_prob)
                down_f.append(filters)
                down_block += [down_cell]
                c_in0, c_in1 = c_in1, c_curr

        num_filters.append(down_f)
        self.blocks += [down_block]

        for i in range(1, self._depth):
            up_f = []
            up_block = nn.ModuleList()
            for j in range(self._depth - i):
                gamma_idx = sum(range(i + j)) + j
                if i + j < self._depth - 1 and self.gamma[gamma_idx] == 0:
                    filters = [0, 0, 0, 'None']
                    up_cell = None
                else:
                    _, _, head_curr, _ = num_filters[0][j]
                    _, _, head_down, _ = num_filters[i - 1][j + 1]
                    head_in0 = sum([num_filters[k][j][2] for k in range(i)])  # up_cell._multiplier
                    head_in1 = head_down  # up_cell._multiplier
                    filters = [head_in0, head_in1, head_curr, 'up']
                    up_cell = BuildCell(genotype, double_down, head_in0, head_in1, head_curr, cell_type='up',
                                        dropout_prob=dropout_prob)
                up_f.append(filters)
                up_block += [up_cell]
            num_filters.append(up_f)
            self.blocks += [up_block]

        self.head_block = nn.ModuleList()

        c_in0 = c
        c_in1 = num_filters[-1][0][2]
        self.head_block.append(Head(genotype, double_down, c_in0, c_in1, nclass))

    def forward(self, x):
        cell_out = []
        for j, cell in enumerate(self.blocks[0]):
            if j == 0:
                # stem0: 1x256x256 -> 32x256x256
                s0 = self.stem0(x)
                # stem1: 32x256x256 -> 32x128x128
                ot = cell(s0)
                cell_out.append(ot)
            elif j == 1:
                ot = cell(s0, cell_out[-1])
                cell_out.append(ot)
            else:
                ot = cell(cell_out[-2], cell_out[-1])
                cell_out.append(ot)

        for j in reversed(range(self._depth - 1)):
            for i in range(1, self._depth - j):
                gamma_idx = sum(range(i + j)) + j
                if i + j < self._depth - 1 and self.gamma[gamma_idx] == 0:
                    ot = None
                    cell_out[i + j] = ot
                else:
                    ides = range(j, i + j)
                    in0 = torch.cat([cell_out[idx] for idx in ides if cell_out[idx] is not None], dim=1)
                    in1 = cell_out[i + j]
                    cell = self.blocks[i][j]
                    ot = cell(in0, in1)
                    cell_out[i + j] = ot

        if self._supervision:
            return [self.head_block[-1](s0, ot) for ot in cell_out]
        else:
            return [self.head_block[-1](s0, cell_out[-1])]
