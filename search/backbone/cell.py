from enum import Enum

from torch.functional import F

from util.genotype import CellLinkDownPos, CellLinkUpPos, CellPos
from util.prim_ops_set import *


class MixedOp(nn.Module):

    def __init__(self, c, stride, op_type):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._op_type = op_type
        self.k = 4
        self.mp = nn.MaxPool2d(2, 2)

        for pri in self._op_type.value:
            op = OPS[pri](c // self.k, stride, affine=False, dp=0)
            self._ops.append(op)

    def forward(self, x, weights_norm, weights_chg):
        # weights: i * 1 where i is the number of normal primitive operations
        # weights: j * 1 where j is the number of up primitive operations
        # weights: k * 1 where k is the number of down primitive operations

        # channel proportion k
        dim_2 = x.shape[1]
        # < 1/k
        xtemp1 = x[:, :  dim_2 // self.k, :, :]
        # > 1/k
        xtemp2 = x[:, dim_2 // self.k:, :, :]

        # down cell needs pooling before concat
        # up cell needs interpolate before concat
        if OpType.UP == self._op_type or OpType.DOWN == self._op_type:
            temp1 = sum(w * op(xtemp1) for w, op in zip(weights_chg, self._ops))
            if OpType.DOWN == self._op_type:
                temp2 = self.mp(xtemp2)
            else:
                temp2 = F.interpolate(xtemp2, scale_factor=2, mode='nearest')
        elif OpType.NORM == self._op_type:
            temp1 = sum(w * op(xtemp1) for w, op in zip(weights_norm, self._ops))
            temp2 = xtemp2
        else:
            raise NotImplementedError()

        ans = torch.cat([temp1, temp2], dim=1)
        return channel_shuffle(ans, self.k)


class Cell(nn.Module):

    def __init__(self, meta_node_num, c_prev_prev, c_prev, c, cell_type):
        super(Cell, self).__init__()
        self.c_prev_prev = c_prev_prev
        self.c_prev = c_prev
        self.c = c
        self._meta_node_num = meta_node_num
        self._multiplier = meta_node_num
        self._input_node_num = 2

        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, stride=2, affine=False,
                                       ops_order='act_weight_norm')
        else:
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, affine=False, ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c_prev, c, kernel_size=1, affine=False, ops_order='act_weight_norm')

        self._ops = nn.ModuleList()

        idx_up_or_down_start = 0 if cell_type == 'down' else 1
        # i=0  j=0,1
        # i=1  j=0,1,2
        # i=2  j=0,1,2,3
        # _ops=2+3+4=9
        for i in range(self._meta_node_num):
            for j in range(self._input_node_num + i):  # the input id for remaining meta-node
                # only the first input is reduction
                # down cell: |_|_|_|_|*|_|_|*|*| where _ indicate down operation
                # up cell:   |*|_|*|*|_|*|_|*|*| where _ indicate up operation
                if idx_up_or_down_start <= j < 2:
                    if cell_type == 'up':
                        op = MixedOp(c, stride=2, op_type=OpType.UP)
                    else:
                        op = MixedOp(c, stride=2, op_type=OpType.DOWN)
                else:
                    op = MixedOp(c, stride=1, op_type=OpType.NORM)

                self._ops.append(op)

    def forward(self, s0, s1, weights_norm, weights_chg, betas):
        # weight1: the normal operations weights with sharing
        # weight2: the down or up operations weight, respectively

        # the cell output is concatenate, so need a convolution to learn best combination
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0

        # offset=0  states=2  _ops=[0,1]
        # offset=2  states=3  _ops=[2,3,4]
        # offset=5  states=4  _ops=[5,6,7,8]
        for i in range(self._meta_node_num):
            # handle the un-consistent dimension
            tmp_list = []
            betas_path = F.softmax(betas[offset:(offset + len(states))], dim=-1)
            for j, h in enumerate(states):
                tmp_list += [
                    betas_path[j] * self._ops[offset + j](h, weights_norm[offset + j], weights_chg[offset + j])]

            # TODO
            # for ss in tmp_list:
            #     print(ss.size())
            # print('-----')

            s = sum(tmp_list)
            offset += len(states)
            states.append(s)

        # concat last 3 states
        return torch.cat(states[-self._multiplier:], dim=1)


class OpType(Enum):
    UP = CellLinkUpPos
    DOWN = CellLinkDownPos
    NORM = CellPos


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
