from torchvision.datasets import *

from .ade20k import ADE20KSegmentation
from .base import *
from .bladder import Bladder
from .camvid import CamVid
from .chaos import CHAOS
from .coco import COCOSegmentation
from .heart import Heart
from .minc import MINCDataset
from .monusac import MoNuSAC
from .pancreas import Pancreas
from .pascal_aug import VOCAugSegmentation
from .pascal_voc import VOCSegmentation
from .pcontext import ContextSegmentation
from .promise12 import Promise12
from .spleen import Spleen
from .ultrasound_nerve import UltraNerve

datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'minc': MINCDataset,
    'cifar10': CIFAR10,
    'ultrasound_nerve': UltraNerve,
    'bladder': Bladder,
    'chaos': CHAOS,
    'promise12': Promise12,
    'camvid': CamVid,
    'monusac': MoNuSAC,
    'heart': Heart,
    'pancreas': Pancreas,
    'spleen': Spleen,
}

acronyms = {
    'coco': 'coco',
    'pascal_voc': 'voc',
    'pascal_aug': 'voc',
    'pcontext': 'pcontext',
    'ade20k': 'ade',
    'citys': 'citys',
    'minc': 'minc',
    'cifar10': 'cifar10',
    'ultrasound_nerve': 'ultrasound_nerve',
    'bladder': 'bladder',
    'chaos': 'chaos',
    'promise12': 'promise12',
    'camvid': 'camvid',
    'monusac': 'monusac',
    'heart': 'heart',
    'pancreas': 'pancreas',
    'spleen': 'spleen',
}

dir = '../train_tiny_data/imgseg/'


def get_dataset(name, path=dir, **kwargs):
    return datasets[name.lower()](root=path, **kwargs)
