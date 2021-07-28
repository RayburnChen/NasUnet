from util.prim_ops_set import *
from .fcn import FCNHead
from .base import BaseNet
from util.functional import *
from torch.nn.functional import interpolate


class BuildCell(nn.Module):
    """Build a cell from genotype"""

    def __init__(self, genotype, c_in0, c_in1, c, cell_type, dropout_prob=0):
        super(BuildCell, self).__init__()

        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = ConvOps(c_in0, c, kernel_size=1, stride=2, ops_order='act_weight_norm')
        else:
            self.preprocess0 = ConvOps(c_in0, c, kernel_size=1, ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c_in1, c, kernel_size=1, ops_order='act_weight_norm')

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
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._num_meta_node):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]

            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]

            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


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

        assert depth >= 2, 'depth must >= 2'
        double_down = 2 if self._double_down_channel else 1
        # 192, 192, 64
        c_s0, c_s1 = 2 * self._multiplier * c, 2 * self._multiplier * c
        c_in0, c_in1, c_curr = c_s0, c_s1, c

        self.blocks = nn.ModuleList()
        self.stem0 = ConvOps(in_channels, c_in0, kernel_size=1, ops_order='weight_norm')
        self.stem1 = ConvOps(in_channels, c_in1, kernel_size=3, stride=2, ops_order='weight_norm')

        num_filters = []
        down_f = []
        down_block = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                filters = [1, 1, int(c_in0/self._multiplier), 'stem0']
                down_cell = self.stem0
                down_f.append(filters)
                down_block += [down_cell]
            elif i == 1:
                filters = [1, 1, int(c_in1/self._multiplier), 'stem1']
                down_cell = self.stem1
                down_f.append(filters)
                down_block += [down_cell]
            else:
                c_curr = int(double_down * c_curr)
                filters = [c_in0, c_in1, c_curr, 'down']
                down_cell = BuildCell(genotype, c_in0, c_in1, c_curr, cell_type='down', dropout_prob=dropout_prob)
                down_f.append(filters)
                down_block += [down_cell]
                c_in0, c_in1 = c_in1, self._multiplier * c_curr  # down_cell._multiplier

        num_filters.append(down_f)
        self.blocks += [down_block]

        for i in range(1, depth):
            up_f = []
            up_block = nn.ModuleList()
            for j in range(depth - i):
                _, _, head_curr, _ = num_filters[i - 1][j]
                _, _, head_down, _ = num_filters[i - 1][j + 1]
                # head_in0 = self._multiplier * sum([num_filters[i-1][j][2]])  # up_cell._multiplier
                head_in0 = self._multiplier * sum([num_filters[k][j][2] for k in range(i)])
                head_in1 = self._multiplier * head_down
                filters = [head_in0, head_in1, head_curr, 'up']
                up_cell = BuildCell(genotype, head_in0, head_in1, head_curr, cell_type='up', dropout_prob=dropout_prob)
                up_f.append(filters)
                up_block += [up_cell]
            num_filters.append(up_f)
            self.blocks += [up_block]

        self.head_block = nn.ModuleList()
        if self._supervision:
            for i in range(1, depth):
                c_last = self._multiplier * num_filters[i][0][2]
                self.head_block += [Head(c_last, nclass)]
        else:
            c_last = self._multiplier * num_filters[-1][0][2]
            self.head_block += [Head(c_last, nclass)]

    def forward(self, x):
        cell_out = []
        final_out = []
        for i, block in enumerate(self.blocks):
            for j, cell in enumerate(block):
                if i == 0 and j == 0:
                    ot = cell(x)
                elif i == 0 and j == 1:
                    ot = cell(x)
                elif i == 0:
                    ot = cell(cell_out[-2], cell_out[-1])
                else:
                    # ides = [sum(range(self._depth, self._depth - i+1, -1)) + j]
                    ides = [sum(range(self._depth, self._depth - k, -1)) + j for k in range(i)]
                    in0 = torch.cat([cell_out[idx] for idx in ides], dim=1)
                    in1 = cell_out[ides[-1] + 1]
                    ot = cell(in0, in1)
                    if j == 0 and self._supervision:
                        final_out.append(self.head_block[i - 1](ot))
                cell_out.append(ot)

        if not self._supervision:
            final_out.append(self.head_block[-1](ot))

        del cell_out
        return final_out


def get_nas_unet(dataset='pascal_voc', **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = NasUnet(datasets[dataset.lower()].NUM_CLASS, datasets[dataset.lower()].IN_CHANNELS,
                    **kwargs)
    return model
