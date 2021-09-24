from torch.functional import F
from util.prim_ops_set import *


class MixedOp(nn.Module):

    def __init__(self, c, op_type):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._op_type = op_type
        self.k = 4
        self.mp = nn.MaxPool2d(2, 2)

        for pri in self._op_type.value:
            op = OPS[pri](c // self.k, c // self.k, self._op_type, dp=0)
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

    def __init__(self, meta_node_num, c_in0, c_in1, c, cell_type):
        super(Cell, self).__init__()
        self.c_in0 = c_in0
        self.c_in1 = c_in1
        self.c = c
        self._meta_node_num = meta_node_num
        self._multiplier = meta_node_num
        self._input_node_num = 2
        self._cell_type = cell_type

        if self._cell_type == 'down':
            # Note: the s0 size is twice than s1!
            # self.preprocess0 = ConvOps(c_in0, c, kernel_size=1, stride=2, ops_order='weight_norm')
            # self.preprocess0 = nn.MaxPool2d(3, stride=2, padding=1)  # suppose c_in0 == c
            self.preprocess0 = nn.Sequential(nn.MaxPool2d(3, stride=2, padding=1), ShrinkBlock(c_in0, c, k=1))
        else:
            # self.preprocess0 = ConvGnReLU(c_in0, c, kernel_size=3)
            self.preprocess0 = ShrinkBlock(c_in0, c, k=1)
        # self.preprocess1 = ConvOps(c_in1, c, kernel_size=1, ops_order='weight_norm_act')
        # self.preprocess1 = nn.Identity()  # suppose c_in1 == c
        self.preprocess1 = ShrinkBlock(c_in1, c, k=1)
        self.c_part = self.preprocess1.c_part

        # self.post_process = ConvGnReLU(c * self._meta_node_num, c, kernel_size=3)
        self.post_process = ExpandBlock(self.c_part * self._meta_node_num, c, cell_type=cell_type)

        self._ops = nn.ModuleList()

        idx_start = 0 if self._cell_type == 'down' else 1
        # i=0  j=0,1
        # i=1  j=0,1,2
        # i=2  j=0,1,2,3
        # _ops=2+3+4=9
        for i in range(self._meta_node_num):
            for j in range(self._input_node_num + i):  # the input id for remaining meta-node
                # only the first input is reduction
                # down cell: |_|_|_|_|*|_|_|*|*| where _ indicate down operation
                # up cell:   |*|_|*|*|_|*|_|*|*| where _ indicate up operation
                if idx_start <= j < 2:
                    if self._cell_type == 'up':
                        op = MixedOp(self.c_part, op_type=OpType.UP)
                    else:
                        op = MixedOp(self.c_part, op_type=OpType.DOWN)
                else:
                    op = MixedOp(self.c_part, op_type=OpType.NORM)

                self._ops.append(op)

    def forward(self, in0, in1, weights_norm, weights_chg, betas):
        # weight1: the normal operations weights with sharing
        # weight2: the down or up operations weight, respectively

        # the cell output is concatenate, so need a convolution to learn best combination
        s0 = self.preprocess0(in0)
        s1 = self.preprocess1(in1)
        states = [s0, s1]
        offset = 0

        # offset=0  states=2  _ops=[0,1]
        # offset=2  states=3  _ops=[2,3,4]
        # offset=5  states=4  _ops=[5,6,7,8]
        for i in range(self._meta_node_num):
            # handle the un-consistent dimension
            tmp_list = []
            betas_path = betas[offset:(offset + len(states))]
            for j, h in enumerate(states):
                tmp_list += [
                    betas_path[j] * self._ops[offset + j](h, weights_norm[offset + j], weights_chg[offset + j])]

            s = sum(tmp_list)
            offset += len(states)
            states.append(s)

        return self.post_process(torch.cat(states[-self._multiplier:], dim=1), in1)




