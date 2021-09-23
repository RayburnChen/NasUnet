import logging
import math
import os
import shutil
import subprocess
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn
from torch.nn.parallel._functions import Broadcast
from torchvision.utils import make_grid

from util.encoder_colors import get_mask_pallete
from ptflops import get_model_complexity_info
from torchstat import stat


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def channel_shuffle(x, groups):
    # Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
    batch, c, h, w = x.size()
    c_per_g = c // groups
    # reshape
    x = x.view(batch, groups, c_per_g, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    return x.view(batch, -1, h, w)


class RunScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self.fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU ": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def calc_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    t, h = divmod(h, 24)
    return {'day': t, 'hour': h, 'minute': m, 'second': int(s)}


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(log_dir):
    create_exp_dir(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'run.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger('Nas Seg')
    logger.addHandler(fh)
    return logger


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def get_gpus_memory_info():
    """Get the maximum free usage memory of gpu"""
    rst = subprocess.run('nvidia-smi -q -d Memory', stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    rst = rst.strip().split('\n')
    memory_available = [int(line.split(':')[1].split(' ')[1]) for line in rst if 'Free' in line][::2]
    id_num = int(np.argmax(memory_available))
    return id_num, memory_available


def calc_parameters_count(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6


def create_exp_dir(path, desc='Experiment dir: {}'):
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def one_hot_encoding(inputs, c):
    """
    One-hot encoder: Converts NxHxW label image to NxCxHxW, where each label is stored in a separate channel
    :param inputs: input image (NxHxW)
    :param c: number of channels/labels
    :return: output image  (NxCxHxW)
    """
    assert inputs.dim() == 3
    n, h, w = inputs.size()
    result = torch.zeros((n, c, h, w))
    # torch.Tensor.scatter_(dim, index, src) -> Tensor
    # eg: For 4d tensor
    #    self[i][index[i][j][k][h]][k][h] = src[i][j][k][h]    # if dim == 1
    result.scatter_(1, inputs.unsqueeze(1), 1)
    return result


def broadcast_list(li, device_ids):
    l_copies = Broadcast.apply(device_ids, *li)  # default broadcast not right?
    l_copies = [l_copies[i:i + len(li)]
                for i in range(0, len(l_copies), len(li))]
    return l_copies


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


def store_images(inputs, predicts, target, dataset='promise12'):
    """
    store the test or valid image in tensorboardX images container
    :param inputs:     NxCxHxW
    :param predicts:  NxCxHxW
    :param target:    NxHxW
    :param dataset: data source
    :return:
    """
    n = inputs.shape[0]
    grid_image_list = []
    for i in range(n):
        channel = inputs[i].shape[0]
        pred = torch.max(predicts[i], 0)[1].cpu().numpy()
        mask2s = get_mask_pallete(pred, dataset, channel=channel)
        if channel == 3:  # rgb
            mask2s = torch.from_numpy(np.array(mask2s).transpose([2, 0, 1])).float()
        else:  # gray
            mask2s = torch.from_numpy(np.expand_dims(np.array(mask2s), axis=0)).float()

        gt = target[i].cpu().numpy()
        target2s = get_mask_pallete(gt, dataset, channel=channel)
        if channel == 3:
            target2s = torch.from_numpy(np.array(target2s).transpose([2, 0, 1])).float()
        else:
            target2s = torch.from_numpy(np.expand_dims(np.array(target2s), axis=0)).float()

        grid_image_list += [inputs[i].cpu(), mask2s, target2s]
    grid_image = make_grid(grid_image_list, normalize=True, scale_each=True)
    return grid_image


def resize_pred_to_val(y_pred, shape):
    """
    :param y_pred: a list of numpy array: [n,h,w]
    :param shape: resize y_pred to [n x h_new x w_new]
    :return: a list of numpy array: [n x h_new x w_new]
    """
    row = shape[1]
    col = shape[2]
    resized_pred = np.zeros(shape)
    for mm in range(len(y_pred)):
        resized_pred[mm, :, :] = cv2.resize(y_pred[mm, :, :, 0], (row, col), interpolation=cv2.INTER_NEAREST)

    return resized_pred.astype(int)


# labels_dict : {ind_label: count_label}
# mu : parameter to tune
def create_class_weight(list_weight, mu=0.15):
    total = np.sum(list_weight)
    new_weight = []
    for weight in list_weight:
        score = math.log(mu * total / float(weight))
        weight = score if score > 1.0 else 1.0
        new_weight += [weight]

    return new_weight


def gpu_memory(n=0):
    t = torch.cuda.get_device_properties(n).total_memory
    r = torch.cuda.memory_reserved(n)
    a = torch.cuda.memory_allocated(n)
    f = r - a  # free inside reserved
    res = t, r, a, f
    return [str(round(x / 1024 / 1024 / 1024, 2)) + ' GB' for x in res]


def complexity_info(model, input_size):
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False,
                                             verbose=True)
    return macs, params


def stat_info(model, input_size):
    stat(model, input_size)
