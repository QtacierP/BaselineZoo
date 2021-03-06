# -*- coding: utf-8 -*-
def build_model(config):
    if 'deeplab' in config.model.model_name or 'resnext' in config.model.model_name:
        from baseline_zoo.segmentation.model.deeplab import DeepLab
        return DeepLab(config=config)
    elif 'unet' in config.model.model_name or 'resnext' in config.model.model_name:
        from baseline_zoo.segmentation.model.unet import UNet
        return UNet(config=config)
    else:
        raise NotImplemented