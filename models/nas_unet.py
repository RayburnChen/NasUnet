from util.prim_ops_set import *
from .fcn import FCNHead
from .base import BaseNet
from util.functional import *
from torch.nn.functional import interpolate


class BuildCell(nn.Module):
    """Build a cell from genotype"""

    def __init__(self, genotype, c_prev_prev, c_prev, c, cell_type, dropout_prob=0):
        super(BuildCell, self).__init__()

        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, stride=2, ops_order='act_weight_norm')
        else:
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c_prev, c, kernel_size=1, ops_order='act_weight_norm')

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
                 genotype=None, double_down_channel=False):

        super(NasUnet, self).__init__(nclass, aux, backbone, norm_layer=nn.GroupNorm)
        self._depth = depth
        self._double_down_channel = double_down_channel
        stem_multiplier = 4
        c_curr = stem_multiplier * c

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, c

        # the stem need a complicate mode
        self.stem0 = ConvOps(in_channels, c_prev_prev, kernel_size=1, ops_order='weight_norm')
        self.stem1 = ConvOps(in_channels, c_prev, kernel_size=3, stride=2, ops_order='weight_norm')

        assert depth >= 2, 'depth must >= 2'

        self.num_filters = []
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            sub_path = []
            up_blocks = nn.ModuleList()
            c_curr = int(2 * c_curr) if self._double_down_channel else c_curr  # double the number of filters
            filters = [c_prev_prev, c_prev, c_curr, 'down']
            down_cell = BuildCell(genotype, c_prev_prev, c_prev, c_curr, cell_type='down', dropout_prob=dropout_prob)
            sub_path.append(filters)
            up_blocks += [down_cell]
            c_prev_prev, c_prev = c_prev, 3 * c_curr  # down_cell._multiplier
            self.num_filters.append(sub_path)
            self.down_blocks += [up_blocks]

        for i in range(depth):
            head_prev_prev, head_prev, head_curr, head_type = self.num_filters[i][0]
            for j in range(i):
                head_prev = 3 * head_curr  # up_cell._multiplier
                head_curr = int(
                    head_curr // 2) if self._double_down_channel else head_curr  # halve the number of filters
                head_prev_prev = 3 * head_curr
                filters = [head_prev_prev, head_prev, head_curr, 'up']
                up_cell = BuildCell(genotype, head_prev_prev, head_prev, head_curr, cell_type='up',
                                    dropout_prob=dropout_prob)
                self.num_filters[i].append(filters)
                self.down_blocks[i] += [up_cell]

        last_filters = 3 * self.num_filters[-1][-1][2]
        self.nas_unet_head = ConvOps(last_filters, nclass, kernel_size=1, ops_order='weight')

        if self.aux:
            self.auxlayer = FCNHead(c_prev, nclass, nn.BatchNorm2d)

    def forward(self, x):
        _, _, h, w = x.size()
        s0, s1 = self.stem0(x), self.stem1(x)
        for i, up_blocks in enumerate(self.down_blocks):
            for j, cell in enumerate(up_blocks):
                if j == 0:
                    in0, in1 = s0, s1
                else:
                    in0, in1 = self.num_filters[i-1][j-1][4], self.num_filters[i][j-1][4]
                ot = cell(in0, in1)
                self.num_filters[i][j].append(ot)
            s0, s1 = s1, self.num_filters[i][0][4]


        output = self.nas_unet_head(self.num_filters[-1][-1][4])

        outputs = []
        outputs.append(output)

        if self.aux:  # use aux header
            auxout = self.auxlayer(self.num_filters[-1][-1][4])
            auxout = interpolate(auxout, (h, w), **self._up_kwargs)
            outputs.append(auxout)

        return outputs


def get_nas_unet(dataset='pascal_voc', **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = NasUnet(datasets[dataset.lower()].NUM_CLASS, datasets[dataset.lower()].IN_CHANNELS,
                    **kwargs)
    return model
