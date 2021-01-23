data_configs = {
    'dataset_type': 'cifar',
    'data_dir': './dataset/cifar10',
    'task': 'classification',
    'dataset': 'cifar10',
    'std': [0.5, 0.5, 0.5],
    'mean': [0.5, 0.5, 0.5],
    'n_colors': 3,
    'n_classes': 10,
    'train_augmentations':  {'resize': {'size': (28, 28)}},
    'test_augmentations': {'resize': {'size': (28, 28)}},
    'size': (28, 28)
}

model_configs = {
    'model_dir': None,
    'model_name': None,
    'n_layers': None,
    'n_downs': None,
    'pre_trained': None,
    'sync_bn': False
}

train_configs = {
    'log_dir': None,
    'optimizer': ('adam', {}),
    'schduler': None,
    'lr': None,
    'epochs': None,
    'warmup_epochs': None,
    'save_freq': None,
    'batch_size_per_gpu': None,
    'callbacks': [],
    'log': None, # List or str
    'val_freq': None, 
    'precision': 32,
    'num_workers': 8,
    'loss': None,
    'metrics': None
}

