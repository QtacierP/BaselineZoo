def build_model(config, args):
    if config.data.task == 'classification':
        from baseline_zoo.classification.model.builder import build_model
    else:
        raise NotImplementedError
    return build_model(config)
    