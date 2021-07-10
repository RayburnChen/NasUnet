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

            # the size of h1 and h2 may be different, so we need interpolate
            # if h1.size() != h2.size() :
            #     _, _, height1, width1 = h1.size()
            #     _, _, height2, width2 = h2.size()
            #     if height1 > height2 or width1 > width2:
            #         h2 = interpolate(h2, (height1, width1))
            #     else:
            #         h1 = interpolate(h1, (height2, width2))
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


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
        c_in0, c_in1, c_curr = self._multiplier * c, self._multiplier * c, c

        self.blocks = nn.ModuleList()
        self.stem0 = ConvOps(in_channels, c_in0, kernel_size=1, ops_order='weight_norm')
        self.stem1 = ConvOps(in_channels, c_in1, kernel_size=3, stride=2, use_transpose=True, ops_order='weight_norm')

        num_filters = []
        down_f = []
        down_block = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                filters = [in_channels, in_channels, c_curr, 'stem0']
                down_cell = self.stem0
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
            c_curr = int(c_curr / double_down)
            c_in0, c_in1 = self._multiplier * num_filters[0][-i-1][2], self._multiplier * num_filters[-1][-1][2]
            filters = [c_in0, c_in1, c_curr, 'up']
            up_cell = BuildCell(genotype, c_in0, c_in1, c_curr, cell_type='up', dropout_prob=dropout_prob)
            up_f.append(filters)
            up_block += [up_cell]
            num_filters.append(up_f)
            self.blocks += [up_block]

        last_filters = self._multiplier * num_filters[-1][-1][2]
        self.nas_unet_head = ConvOps(last_filters, nclass, kernel_size=1, ops_order='weight')

    def forward(self, x):
        cell_out = []
        final_out = []
        for i, block in enumerate(self.blocks):
            for j, cell in enumerate(block):
                if i == 0 and j == 0:
                    ot = cell(x)
                elif i == 0 and j == 1:
                    ot = cell(self.stem1(x), cell_out[j - 1])
                elif i == 0:
                    ot = cell(cell_out[j - 2], cell_out[j - 1])
                else:
                    in0 = cell_out[self._depth-1-i]
                    in1 = cell_out[-1]
                    ot = cell(in0, in1)
                    if j == 0 and self._supervision:
                        final_out.append(self.nas_unet_head(ot))
                cell_out.append(ot)

        if not self._supervision:
            final_out.append(self.nas_unet_head(cell_out[-1]))

        del cell_out
        return final_out


def get_nas_unet(dataset='pascal_voc', **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = NasUnet(datasets[dataset.lower()].NUM_CLASS, datasets[dataset.lower()].IN_CHANNELS,
                    **kwargs)
    return model
