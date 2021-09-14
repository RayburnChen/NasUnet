import torch
from torch.functional import F

from models import BasicBlock
from util.prim_ops_set import *
from util.genotype import *
from search.backbone.cell import Cell



class Head(nn.Module):

    def __init__(self, c_last, nclass):
        super(Head, self).__init__()
        self.head = ConvOps(c_last, nclass, kernel_size=1, ops_order='weight')

    def forward(self, ot):
        return self.head(ot)


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
        c_in0, c_in1, c_curr = c, c, c

        self.blocks = nn.ModuleList()
        self.stem0 = ConvOps(in_channels, c_in0, kernel_size=1, ops_order='weight_norm_act')
        skip_down = ConvOps(c_in0, c_in1, kernel_size=1, stride=2, ops_order='weight_norm')
        self.stem1 = BasicBlock(c_in0, c_in1, stride=2, dilation=1, downsample=skip_down, previous_dilation=1,
                                norm_layer=nn.BatchNorm2d)

        num_filters = []
        down_f = []
        down_block = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                filters = [1, 1, int(c_in0), 'stem0']
                down_cell = self.stem0
                down_f.append(filters)
                down_block += [down_cell]
            elif i == 1:
                filters = [1, 1, int(c_in1), 'stem1']
                down_cell = self.stem1
                down_f.append(filters)
                down_block += [down_cell]
            else:
                c_curr = int(double_down * c_curr)
                filters = [c_in0, c_in1, c_curr, 'down']
                down_cell = Cell(meta_node_num, c_in0, c_in1, c_curr, cell_type='down')
                down_f.append(filters)
                down_block += [down_cell]
                c_in0, c_in1 = c_in1, c_curr  # down_cell._multiplier

        num_filters.append(down_f)
        self.blocks += [down_block]

        for i in range(1, depth):
            up_f = []
            up_block = nn.ModuleList()
            for j in range(depth - i):
                _, _, head_curr, _ = num_filters[i - 1][j]
                _, _, head_down, _ = num_filters[i - 1][j + 1]
                head_in0 = sum([num_filters[k][j][2] for k in range(i)])  # up_cell._multiplier
                head_in1 = head_down  # up_cell._multiplier
                filters = [head_in0, head_in1, head_curr, 'up']
                up_cell = Cell(meta_node_num, head_in0, head_in1, head_curr, cell_type='up')
                up_f.append(filters)
                up_block += [up_cell]
            num_filters.append(up_f)
            self.blocks += [up_block]

        self.head_block = nn.ModuleList()

        c_last = num_filters[-1][0][2]
        self.head_block += [Head(c_last, nclass)]

        if use_softmax_head:
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, weights_down_norm, weights_up_norm, weights_down, weights_up, betas_down, betas_up, gamma):
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
                    ot = cell(cell_out[-2], cell_out[-1], weights_down_norm, weights_down, betas_down)
                else:
                    ides = [sum(range(self._depth, self._depth - k, -1)) + j for k in range(i)]
                    # in0 = torch.cat([cell_out[idx] for idx in ides], dim=1)
                    # gamma_used = gamma[j:i + j, :]
                    # assert len(ides) == len(gamma_used), 'gamma size not match'
                    # in0 = torch.cat(
                    #     [cell_out[ides[0]] * gam[0] + cell_out[idx] * gam[1] for idx, gam in zip(ides, gamma_used)],
                    #     dim=1)
                    # in0 = torch.cat(
                    #     [cell_out[ides[i - 1 if i > 0 else 0]] * gamma_used[i][0] + cell_out[idx] * gamma_used[i][1] for
                    #      i, idx in enumerate(ides)], dim=1)
                    in0 = cell_out[ides[0]]

                    if i > 1:
                        gamma_ides = [sum(range(self._depth - 2, self._depth - 2 - k, -1)) + j for k in range(i - 1)]
                        in0 = torch.cat(
                            [in0] + [cell_out[ides[k]] * gamma[idx][0] + cell_out[ides[k + 1]] * gamma[idx][1] for
                                     k, idx in enumerate(gamma_ides)], dim=1)
                    in1 = cell_out[ides[-1] + 1]
                    ot = cell(in0, in1, weights_up_norm, weights_up, betas_up)
                    # if i + j < self._depth - 1:
                    #     ot = gamma[i + j][0] * cell_out[ides[-1]] + gamma[i + j][1] * ot
                    if j == 0 and self._supervision:
                        final_out.append(self.head_block[-1](ot))
                cell_out.append(ot)

        del cell_out
        if self._supervision:
            return final_out
        else:
            return [self.head_block[-1](ot)]


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
        self._init_alphas(depth)

    def _init_alphas(self, depth):

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

        self.gamma = nn.Parameter(1e-3 * torch.randn(sum(range(depth - 1)), 2))

        self._arch_parameters = [
            self.alphas_down,
            self.alphas_up,
            self.alphas_normal_down,
            self.alphas_normal_up,
            self.betas_down,
            self.betas_up,
            self.gamma
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
        betas_down = []
        betas_up = []
        for i in range(self._meta_node_num):
            offset = len(betas_down)
            betas_down.append(F.softmax(self.betas_down[offset:offset + 2 + i], dim=-1).detach().cpu())
            betas_up.append(F.softmax(self.betas_up[offset:offset + 2 + i], dim=-1).detach().cpu())
        betas_down = torch.cat(betas_down, dim=0)
        betas_up = torch.cat(betas_up, dim=0)

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
        gamma = F.softmax(self.gamma, dim=-1).detach().cpu().argmax(1).tolist()
        gamma[0] = 1
        geno_type = Genotype(
            down=gene_down, down_concat=concat,
            up=gene_up, up_concat=concat,
            gamma=gamma
        )
        return geno_type

    def forward(self, x):

        alphas_down_norm = F.softmax(self.alphas_normal_down, dim=-1)
        alphas_up_norm = F.softmax(self.alphas_normal_up, dim=-1)
        alphas_down = F.softmax(self.alphas_down, dim=-1)
        alphas_up = F.softmax(self.alphas_up, dim=-1)
        betas_down = []
        betas_up = []
        for i in range(self._meta_node_num):
            offset = len(betas_down)
            betas_down.append(F.softmax(self.betas_down[offset:offset + 2 + i], dim=-1))
            betas_up.append(F.softmax(self.betas_up[offset:offset + 2 + i], dim=-1))
        betas_down = torch.cat(betas_down, dim=0)
        betas_up = torch.cat(betas_up, dim=0)
        gamma = F.softmax(self.gamma, dim=-1)

        if len(self.device_ids) == 1:
            return self.net(x, alphas_down_norm, alphas_up_norm, alphas_down, alphas_up, betas_down, betas_up, gamma)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_down_copies = broadcast_list(alphas_down_norm, self.device_ids)
        wnormal_up_copies = broadcast_list(alphas_up_norm, self.device_ids)
        wdown_copies = broadcast_list(alphas_down, self.device_ids)
        wup_copies = broadcast_list(alphas_up, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas, list(zip(xs, wnormal_down_copies, wnormal_up_copies,
                                                                wdown_copies, wup_copies)),
                                             devices=self.device_ids)

        return nn.parallel.gather(outputs, self.device_ids[0])


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
