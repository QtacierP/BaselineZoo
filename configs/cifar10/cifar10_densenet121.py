from configs.cifar10.cifar10_base import data_configs

model_configs = {
    'model_dir': './experiment/cifar10/densenet121',
    'model_name': 'densenet121',
    'n_layers': None,
    'n_downs': None,
    'pre_trained': False,
    'sync_bn': False
}

train_configs = {
    'log_dir': './experimrnt/cifar10/densenet121/logs/',
    'optimizer': ('adam', {}),
    'scheduler': ('cosine', {'T_max': 50, 'eta_min': 0}),
    'lr': 0.02,
    'epochs': 50,
    'warmup_epochs': 0,
    'save_freq': 10,
    'batch_size_per_gpu': 256,
    'callbacks': {'ModelCheckpoint'  : {'filepath': './experiment/cifar10/densenet121/models/', 'monitor': 'val_loss'},
                  'ProgressBar'      : {'logging_interval': 'step'}},
    'log': {'TensorBoardLogger': {'save_dir': './experiment/cifar10/densenet121/logs/'}}, 
    'val_freq': 1,
    'precision': 32,
    'num_workers': 8,
    'loss': 'ce',
    'metrics': 'accuracy'
}

