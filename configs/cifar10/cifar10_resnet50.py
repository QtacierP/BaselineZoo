from configs.cifar10.cifar10_base import data_configs

model_configs = {
    'model_dir': './experiment/cifar10/resnet50',
    'model_name': 'resnet50',
    'n_layers': None,
    'n_downs': None,
    'pre_trained': False,
    'sync_bn': False
}

train_configs = {
    'log_dir': './experimrnt/cifar10/resnet50/logs/',
    'optimizer': ('adam', {}),
    'scheduler': ('cosine', {'T_max': 50, 'eta_min': 0}),
    'lr': 0.02,
    'epochs': 50,
    'warmup_epochs': 0,
    'save_freq': 10,
    'batch_size_per_gpu': 8,
    'callbacks': {'ModelCheckpoint'  : {'filepath': './experiment/cifar10/resnet50/models/', 'monitor': 'val_loss'},
                  'ProgressBar'      : {}},
    'log': {'TensorBoardLogger': {'save_dir': './experiment/cifar10/resnet50/logs/'}}, 
    'val_freq': 1,
    'precision': 32,
    'num_workers': 8,
    'loss': 'ce',
    'metrics': 'accuracy'
}

