import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLosses(nn.Module):

    def __init__(self, name='dice_ce'):
        super(SegmentationLosses, self).__init__()
        print('Using loss: {}'.format(name))
        if name == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        elif name == 'dice_ce':
            self.loss = DiceCrossEntropyLoss()
        elif name == 'dice_loss':
            self.loss = SoftDiceLoss()
        else:
            raise NotImplementedError
        self.smooth = np.spacing(1)

    def forward(self, inputs, targets):
        return self.loss.forward(inputs, targets)

    # https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    def _dice_loss(self, inputs, targets):
        predict = F.softmax(inputs, dim=1)[:, 1:].squeeze(1)  # BATCH x H x W

        intersection = (predict * targets).float().sum((1, 2))
        sum_predict = predict.float().sum((1, 2))
        sum_target = targets.float().sum((1, 2))

        dice = (2. * intersection + self.smooth) / (
                sum_predict + sum_target + self.smooth)  # We smooth our devision to avoid 0/0
        return 1 - dice.mean()

    def _iou_loss(self, inputs, targets):
        predict = F.softmax(inputs, dim=1)[:, 1:].squeeze(1)  # BATCH x H x W

        intersection = (predict * targets).float().sum((1, 2))
        total = predict.float().sum((1, 2)) + targets.float().sum((1, 2))
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

    def _bce_loss(self, inputs, targets):
        prob = F.softmax(inputs, dim=1)[:, 1:].squeeze(1)
        return F.binary_cross_entropy(prob, targets.float(), reduction='mean')


class SoftDiceLoss(nn.Module):
    def __init__(self, do_bg=False, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        axes = [0] + list(range(2, len(shp_x)))

        x = F.softmax(x, 1)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            dc = dc[1:]

        dc = dc.mean()

        return -dc


class SoftDiceLossSquared(nn.Module):
    def __init__(self, do_bg=False, smooth=1e-5):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        axes = [0] + list(range(2, len(shp_x)))

        x = F.softmax(x, 1)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            dc = dc[1:]

        dc = dc.mean()

        return -dc


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, square_dice=False, weight_ce=1, weight_dice=1, log_dice=False):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DiceCrossEntropyLoss, self).__init__()
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss()

        if not square_dice:
            self.dc = SoftDiceLoss()
        else:
            self.dc = SoftDiceLossSquared()

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target.long()) if self.weight_ce != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss

        return result


def sum_tensor(inp, axes, keep_dim=False):
    axes = np.unique(axes).astype(int)
    if keep_dim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keep_dim=False)
        fp = sum_tensor(fp, axes, keep_dim=False)
        fn = sum_tensor(fn, axes, keep_dim=False)
        tn = sum_tensor(tn, axes, keep_dim=False)

    return tp, fp, fn, tn
