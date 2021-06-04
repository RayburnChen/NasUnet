import threading
import torch
import numpy as np
import torch.nn.functional as F

SMOOTH = np.spacing(1)


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes"""

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            mean_acc = mean_pix_accuracy(pred, label)
            mean_iou = mean_intersection_union(pred, label, self.nclass)
            with self.lock:
                self.total_mean_acc += mean_acc
                self.count_mean_acc += 1
                self.total_mean_iou += mean_iou
                self.count_mean_iou += 1
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get(self):
        pixAcc = 100.0 * self.total_mean_acc / self.count_mean_acc
        mIoU = 100.0 * self.total_mean_iou / self.count_mean_iou
        return pixAcc, mIoU

    def reset(self):
        self.total_mean_iou = 0
        self.count_mean_iou = 0
        self.total_mean_acc = 0
        self.count_mean_acc = 0
        return


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


# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy

def mean_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    predict = torch.max(output, dim=1)[1]  # BATCH x H x W

    intersection = (predict & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (predict | target).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.mean()


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image. 
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))

    return pixel_correct, pixel_labeled


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def rel_abs_vol_diff(y_true, y_pred):
    return np.abs((y_pred.sum() / y_true.sum() - 1) * 100)


def get_boundary(data, img_dim=2, shift=-1):
    data = data > 0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data, shift=shift, axis=nn))
    return edge.astype(int)


def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):
    intersection = y_true * y_pred
    return (2. * intersection.sum(axis=axis) + smooth) / (
            np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + smooth)


def dice_coefficient(inputs, target):

    predict = torch.argmax(inputs, dim=1)  # BATCH x H x W

    intersection = (predict & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    den1 = predict.float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    den2 = target.float().sum((1, 2))  # Will be zzero if both are 0

    dice = (2.*intersection + SMOOTH) / (den1 + den2 + SMOOTH)  # We smooth our devision to avoid 0/0

    return dice.mean()
