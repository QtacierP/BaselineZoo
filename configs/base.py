data_configs = {
    'dataset_type': None,
    'data_dir': None,
    'task': None,
    'dataset': None,
    'std': None,
    'mean': None,
    'n_colors': None,
    'n_classes': None,
    'train_augmentations':  {},
    'test_augmentations': {},
    'size': ()
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
    'optimizer': None,
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

