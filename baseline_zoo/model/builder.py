def build_model(config):
    if 'resnet' in config.model.model_name:
        from baseline_zoo.model.resnet import ResNet
        return ResNet(config=config)
