# -*- coding: utf-8 -*-
def build_model(config):
    if 'resnet' in config.model.model_name or 'resnext' in config.model.model_name:
        from baseline_zoo.classification.model.resnet import ResNet
        return ResNet(config=config)
    elif 'densenet' in config.model.model_name:
        from baseline_zoo.classification.model.densenet import DenseNet
        return DenseNet(config=config)
