data_configs = {
    'dataset_type': 'fold-segmentation',
    'data_dir': './dataset/drive/',
    'task': 'segmentation',
    'dataset': 'drive',
    'std': [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255],
    'mean': [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255],
    'n_colors': 3,
    'n_classes': 2,
    'train_augmentations':  {'resize': {'size': (512, 512)},
                             'horizontal_flip': {}, 
                             'affine': {'degrees':  30}, 
                             'color_jitter': {'brightness': 0.3, 'contrast': 0.3, 
                                              'saturation': 0.3, 'hue': 0.3}},
    'test_augmentations': {'resize': {'size': (512, 512)}},
    'size': (512, 512)
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
    'num_workers': 8
}

