data_configs = {
    'data_dir': './dataset/cifar10',
    'task': 'classification',
    'dataset': 'cifar10',
    'std': [0.5, 0.5, 0.5],
    'mean': [0.5, 0.5, 0.5],
    'n_colors': 3,
    'n_classes': 10,
    'train_augmentations':  {},
    'test_augmentations': {},
}

model_configs = {
    'model_dir': None,
    'model_name': None,
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