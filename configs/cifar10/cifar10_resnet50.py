from configs.cifar10.cifar10_base import data_configs

model_configs = {
    'model_dir': './experiment/cifar10/resnet50',
    'model_name': 'resnet50',
    'n_layers': None,
    'n_downs': None,
    'pre_trained': None
}

train_configs = {
    'log_dir': None,
    'optimizer': None,
    'schduler': None,
    'lr': None,
    'epochs': None,
    'warmup_epochs': None,
    'save_freq': None,
    'batch_size_per_gpu': None
}