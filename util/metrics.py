import threading
import torch
import numpy as np
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor

SMOOTH = np.spacing(1)


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes"""

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.acc = AverageMeter()
        self.tp_list = []
        self.fp_list = []
        self.fn_list = []
        self.reset()

    def evaluate_worker(self, label, pred):
        mean_acc = mean_pix_accuracy(pred, label)
        tp, fp, fn = confusion_matrix(pred, label)
        with self.lock:
            self.acc.update(mean_acc)
            self.tp_list.append([tp])
            self.fp_list.append([fp])
            self.fn_list.append([fn])
        return

    def update(self, labels, preds):
        if isinstance(preds, torch.Tensor):
            self.evaluate_worker(labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=self.evaluate_worker,
                                        args=(label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get(self):
        pixAcc = self.acc.mperc()
        mIoU = percentage(self.miou())
        dice = percentage(self.dice())
        return pixAcc, mIoU, dice

    def miou(self):
        tp_total = np.sum(self.tp_list, 0)
        fp_total = np.sum(self.fp_list, 0)
        fn_total = np.sum(self.fn_list, 0)
        return (tp_total + SMOOTH) / (tp_total + fp_total + fn_total + SMOOTH)

    def dice(self):
        tp_total = np.sum(self.tp_list, 0)
        fp_total = np.sum(self.fp_list, 0)
        fn_total = np.sum(self.fn_list, 0)
        return (2 * tp_total + SMOOTH) / (2 * tp_total + fp_total + fn_total + SMOOTH)

    def reset(self):
        self.acc.reset()
        self.tp_list = []
        self.fp_list = []
        self.fn_list = []
        return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def mloss(self):
        return self.avg

    def mperc(self):
        return percentage(self.avg)


def percentage(value, dec=3):
    if isinstance(value, Tensor):
        value = value.item()
    if isinstance(value, ndarray):
        value = np.mean(value)
    return round(100.0 * value, dec)


def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    predict = torch.max(output, 1)[1]

    # label: 0, 1, ..., nclass - 1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def mean_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    predict = torch.max(output, dim=1)[1]  # BATCH x H x W

    # label: 0, 1, ..., nclass - 1
    # Note: 0 is background
    pixel_labeled = (target > 0).float().sum((1, 2))
    pixel_correct = (predict & (target > 0)).float().sum((1, 2))

    pix_acc = (pixel_correct + SMOOTH) / (pixel_labeled + SMOOTH)

    return pix_acc.mean()


def confusion_matrix(output, label):
    with torch.no_grad():
        num_classes = output.shape[1]
        output_softmax = F.softmax(output, 1)
        output_seg = output_softmax.argmax(1)
        axes = tuple(range(1, len(label.shape)))
        tp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(output_seg.device.index)
        fp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(output_seg.device.index)
        fn_hard = torch.zeros((label.shape[0], num_classes - 1)).to(output_seg.device.index)
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (label == c).float(), axes=axes)
            fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (label != c).float(), axes=axes)
            fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (label == c).float(), axes=axes)

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()
        return tp_hard, fp_hard, fn_hard


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    predict = torch.max(output, 1)[1]
    mini = 1
    maxi = nclass - 1
    nbins = nclass - 1

    # label is: 0, 1, 2, ..., nclass-1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union
