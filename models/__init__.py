from segmentation_models_pytorch import UnetPlusPlus, DeepLabV3, FPN, Linknet, MAnet, PSPNet, Unet, PAN
from .nas_unet import *


def unet(dataset, **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = Unet(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10')
    return model


def unet_plus_plus(dataset, **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = UnetPlusPlus(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10')
    return model


def nasunet(dataset, **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = NasUnet(datasets[dataset.lower()].NUM_CLASS, datasets[dataset.lower()].IN_CHANNELS,
                    **kwargs)
    return model


def deeplab_v3(dataset, **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = DeepLabV3(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10')
    return model


def fpn(dataset, **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = FPN(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10')
    return model


def linknet(dataset, **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = Linknet(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10')
    return model


def manet(dataset, **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = MAnet(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10')
    return model


def pspnet(dataset, **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = PSPNet(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10')
    return model


def pan(dataset, **kwargs):
    # infer number of classes
    from util.datasets import datasets
    model = PAN(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10')
    return model


def get_segmentation_model(name, **kwargs):
    models = {
        'nasunet': nasunet,
        'unet': unet,
        'unet_plus_plus': unet_plus_plus,
        'deeplab_v3': deeplab_v3,
        'fpn': fpn,
        'linknet': linknet,
        'manet': manet,
        'pspnet': pspnet,
        'pan': pan,
    }
    return models[name.lower()](**kwargs)
