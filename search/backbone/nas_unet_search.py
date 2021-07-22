import torch
from torch.functional import F
from util.prim_ops_set import *
from util.genotype import *
from search.backbone.cell import Cell


class Head(nn.Module):

    def __init__(self, c_s0, c_s1, c_last, c_curr, nclass, multiplier):
        super(Head, self).__init__()
        self.tail1 = Cell(multiplier, c_s1, c_last, c_curr, cell_type='up')
        self.tail0 = Cell(multiplier, c_s0, multiplier * c_curr, c_curr, cell_type='up')
        self.head = ConvOps(multiplier * c_curr, nclass, kernel_size=1, ops_order='weight')

    def forward(self, s0, s1, ot, weights_up_norm, weights_up, betas_up):
        return self.head(self.tail0(s0, self.tail1(s1, ot, weights_up_norm, weights_up, betas_up), weights_up_norm, weights_up, betas_up))


class SearchULikeCNN(nn.Module):

    def __init__(self, input_c, c, nclass, depth, meta_node_num=3,
                 double_down_channel=True, use_softmax_head=False, supervision=False):
        super(SearchULikeCNN, self).__init__()
        self._num_classes = nclass  # 2
        self._depth = depth  # 4
        self._meta_node_num = meta_node_num  # 3
        self._multiplier = meta_node_num  # 3
        self._use_softmax_head = use_softmax_head
        self._double_down_channel = double_down_channel
        self._supervision = supervision

        in_channels = input_c
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
            c_curr = int(double_down * c_curr)
            filters = [c_in0, c_in1, c_curr, 'down']
            down_cell = Cell(meta_node_num, c_in0, c_in1, c_curr, cell_type='down')
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
                head_in0 = self._multiplier * sum([num_filters[k][j][2] for k in range(i)])  # up_cell._multiplier
                head_in1 = self._multiplier * head_down  # up_cell._multiplier
                filters = [head_in0, head_in1, head_curr, 'up']
                up_cell = Cell(meta_node_num, head_in0, head_in1, head_curr, cell_type='up')
                up_f.append(filters)
                up_block += [up_cell]
            num_filters.append(up_f)
            self.blocks += [up_block]

        self.head_block = nn.ModuleList()
        for i in range(0, depth if self._supervision else 1):
            c_last = self._multiplier * num_filters[i][0][2]
            self.head_block += [Head(c_s0, c_s1, c_last, c, nclass, self._multiplier)]

        if use_softmax_head:
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, weights_down_norm, weights_up_norm, weights_down, weights_up, betas_down, betas_up):
        s0 = self.stem0(x)
        s1 = self.stem1(x)
        cell_out = []
        final_out = []
        for i, block in enumerate(self.blocks):
            for j, cell in enumerate(block):
                if i == 0 and j == 0:
                    ot = cell(s0, s1, weights_down_norm, weights_down, betas_down)
                elif i == 0 and j == 1:
                    ot = cell(s1, cell_out[-1], weights_down_norm, weights_down, betas_down)
                elif i == 0:
                    ot = cell(cell_out[-2], cell_out[-1], weights_down_norm, weights_down, betas_down)
                else:
                    # ides = [sum(range(self._depth, self._depth - i+1)) + j]
                    ides = [sum(range(self._depth, self._depth - k, -1)) + j for k in range(i)]
                    in0 = torch.cat([cell_out[idx] for idx in ides], dim=1)
                    in1 = cell_out[ides[-1] + 1]
                    ot = cell(in0, in1, weights_up_norm, weights_up, betas_up)
                    if j == 0 and self._supervision:
                        final_out.append(self.head_block[i - 1](s0, s1, ot, weights_up_norm, weights_up, betas_up))
                cell_out.append(ot)

        if not self._supervision:
            final_out.append(self.head_block[-1](s0, s1, ot, weights_up_norm, weights_up, betas_up))

        del cell_out
        return final_out


class NasUnetSearch(nn.Module):

    def __init__(self, input_c, c, num_classes, depth, meta_node_num=4,
                 use_sharing=True, double_down_channel=True, use_softmax_head=False, supervision=False,
                 multi_gpus=False, device='cuda'):
        super(NasUnetSearch, self).__init__()
        self._use_sharing = use_sharing
        self._meta_node_num = meta_node_num

        self.net = SearchULikeCNN(input_c, c, num_classes, depth, meta_node_num,
                                  double_down_channel, use_softmax_head, supervision)

        if 'cuda' == str(device.type) and multi_gpus:
            device_ids = list(range(torch.cuda.device_count()))
            self.device_ids = device_ids
        else:
            self.device_ids = [0]

        # Initialize architecture parameters: alpha
        self._init_alphas()

    def _init_alphas(self):

        normal_num_ops = len(CellPos)
        down_num_ops = len(CellLinkDownPos)
        up_num_ops = len(CellLinkUpPos)

        k = sum(1 for i in range(self._meta_node_num) for n in range(2 + i))  # total number of input node
        self.alphas_down = nn.Parameter(1e-3 * torch.randn(k, down_num_ops))
        self.alphas_up = nn.Parameter(1e-3 * torch.randn(k, up_num_ops))
        self.alphas_normal_down = nn.Parameter(1e-3 * torch.randn(k, normal_num_ops))
        self.alphas_normal_up = self.alphas_normal_down if self._use_sharing else nn.Parameter(
            1e-3 * torch.randn(k, normal_num_ops))

        self.betas_down = nn.Parameter(1e-3 * torch.randn(k))
        self.betas_up = nn.Parameter(1e-3 * torch.randn(k))

        self._arch_parameters = [
            self.alphas_down,
            self.alphas_up,
            self.alphas_normal_down,
            self.alphas_normal_up,
            self.betas_down,
            self.betas_up
        ]

    def load_params(self, alphas_dict, betas_dict):
        self.alphas_down = alphas_dict['alphas_down']
        self.alphas_up = alphas_dict['alphas_up']
        self.alphas_normal_down = alphas_dict['alphas_normal_down']
        self.alphas_normal_up = alphas_dict['alphas_normal_up']
        self.betas_down = betas_dict['betas_down']
        self.betas_up = betas_dict['betas_up']
        self._arch_parameters = [
            self.alphas_down,
            self.alphas_up,
            self.alphas_normal_down,
            self.alphas_normal_up,
            self.betas_down,
            self.betas_up
        ]

    def alphas_dict(self):
        return {
            'alphas_down': self.alphas_down,
            'alphas_normal_down': self.alphas_normal_down,
            'alphas_up': self.alphas_up,
            'alphas_normal_up': self.alphas_normal_up,
        }

    def betas_dict(self):
        return {
            'betas_down': self.betas_down,
            'betas_up': self.betas_up
        }

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        # Note: Since we stack cells by s0: prev prev cells output; s1: prev cells output
        # and when a cell is a up cell, the s0 will be horizontal input and can't do up operation
        # which is different from down cells (s0 and s1 all need down operation). so when
        # parse a up cell string, the string operations is |_|*|_|...|_|, where * indicate up operation
        # mask1 and mask2 below is convenient to handle it.
        alphas_normal_down = F.softmax(self.alphas_normal_down, dim=-1).detach().cpu()
        alphas_down = F.softmax(self.alphas_down, dim=-1).detach().cpu()
        alphas_normal_up = F.softmax(self.alphas_normal_up, dim=-1).detach().cpu()
        alphas_up = F.softmax(self.alphas_up, dim=-1).detach().cpu()
        betas_down = torch.empty(0)
        betas_up = torch.empty(0)
        for i in range(self._meta_node_num):
            offset = len(betas_down)
            betas_down_edge = F.softmax(self.betas_down[offset:offset + 2 + i], dim=-1).detach().cpu()
            betas_up_edge = F.softmax(self.betas_up[offset:offset + 2 + i], dim=-1).detach().cpu()
            betas_down = torch.cat([betas_down, betas_down_edge], dim=0)
            betas_up = torch.cat([betas_up, betas_up_edge], dim=0)

        k = sum(1 for i in range(self._meta_node_num) for n in range(2 + i))  # total number of input node
        for j in range(k):
            alphas_normal_down[j, :] = alphas_normal_down[j, :] * betas_down[j].item()
            alphas_down[j, :] = alphas_down[j, :] * betas_down[j].item()
            alphas_normal_up[j, :] = alphas_normal_up[j, :] * betas_up[j].item()
            alphas_up[j, :] = alphas_up[j, :] * betas_up[j].item()

        geno_parser = GenoParser(self._meta_node_num)
        gene_down = geno_parser.parse(alphas_normal_down.numpy(), alphas_down.numpy(), cell_type='down')
        gene_up = geno_parser.parse(alphas_normal_up.numpy(), alphas_up.numpy(), cell_type='up')
        concat = range(2, self._meta_node_num + 2)
        geno_type = Genotype(
            down=gene_down, down_concat=concat,
            up=gene_up, up_concat=concat
        )
        return geno_type

    def forward(self, x):

        weights_down_norm = F.softmax(self.alphas_normal_down, dim=-1)
        weights_up_norm = F.softmax(self.alphas_normal_up, dim=-1)
        weights_down = F.softmax(self.alphas_down, dim=-1)
        weights_up = F.softmax(self.alphas_up, dim=-1)

        if len(self.device_ids) == 1:
            return self.net(x, weights_down_norm, weights_up_norm, weights_down, weights_up, self.betas_down,
                            self.betas_up)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_down_copies = broadcast_list(weights_down_norm, self.device_ids)
        wnormal_up_copies = broadcast_list(weights_up_norm, self.device_ids)
        wdown_copies = broadcast_list(weights_down, self.device_ids)
        wup_copies = broadcast_list(weights_up, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas, list(zip(xs, wnormal_down_copies, wnormal_up_copies,
                                                                wdown_copies, wup_copies)),
                                             devices=self.device_ids)

        return nn.parallel.gather(outputs, self.device_ids[0])

    # def alphas(self):
    #     for n, p in self._alphas:
    #         yield p

    # def named_alphas(self):
    #     for n, p in self._alphas:
    #         yield n, p


class Architecture(object):

    def __init__(self, model, arch_optimizer, criterion):
        self.model = model
        self.optimizer = arch_optimizer
        self.criterion = criterion

    def step(self, input_valid, target_valid):
        """Do one step of gradient descent for architecture parameters

        Args:
            input_valid: A tensor with N * C * H * W for validation data
            target_valid: A tensor with N * 1 for validation target
            eta:
            network_optimizer:
        """

        self.optimizer.zero_grad()
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)
        loss.backward()
        self.optimizer.step()
