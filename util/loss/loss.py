import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLosses(nn.Module):
    def __init__(self, name='dice_loss', se_loss=False,
                 aux_weight=None, weight=None, ignore_index=0):
        '''2D Cross Entropy Loss with Auxiliary Loss or Dice Loss

        :param name: (string) type of loss : ['dice_loss', 'cross_entropy', 'cross_entropy_with_dice']
        :param aux_weight: (float) weights of an auxiliary layer or the weight of dice loss
        :param weight: (torch.tensor) the weights of each class
        :param ignore_index: (torch.tensor) ignore i class.
        '''
        super(SegmentationLosses, self).__init__()
        self.se_loss = se_loss
        self.name = name
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = True
        self.reduce = True
        self.smooth = np.spacing(1)
        print('Using loss: {}'.format(name))

    def forward(self, inputs, targets):
        if self.name == 'dice_loss':
            return self._dice_loss(inputs, targets)
        elif self.name == 'iou_loss':
            return self._iou_loss(inputs, targets)
        elif self.name == 'bce_loss':
            return self._bce_loss(inputs, targets)
        elif self.name == 'bce_dice_loss':
            return self._bce_loss(inputs, targets) + self._dice_loss(inputs, targets)
        elif self.name == 'bce_iou_loss':
            return self._bce_loss(inputs, targets) + self._iou_loss(inputs, targets)
        elif self.name == 'cross_entropy':
            return F.cross_entropy(inputs, targets)
        else:
            raise NotImplementedError

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

        iou = (intersection + self.smooth)/(union + self.smooth)
        return 1 - iou.mean()

    def _bce_loss(self, inputs, targets):
        prob = F.softmax(inputs, dim=1)[:, 1:].squeeze(1)
        return F.binary_cross_entropy(prob, targets.float(), reduction='mean')
