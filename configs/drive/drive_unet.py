from configs.drive.drive_base import data_configs


model_configs = {
    'model_dir': './experiment/drive/unet',
    'model_name': 'unet',
    'n_layers': None,
    'n_downs': None,
    'pre_trained': False,
    'sync_bn': False
}

train_configs = {
    'log_dir': './experimrnt/drive/unet/logs/',
    'optimizer': ('adam', {}),
    'scheduler': None,
    'lr': 1e-5,
    'epochs': 50,
    'warmup_epochs': 0,
    'save_freq': 10,
    'batch_size_per_gpu': 2,
    'callbacks': {'ModelCheckpoint'  : {'filepath': './experiment/drive/unet/models/', 'monitor': 'val_loss'},
                  'ProgressBar'      : {}},
    'log': {'TensorBoardLogger': {'save_dir': './experiment/drive/unet/logs/'}}, 
    'val_freq': 1,
    'precision': 32,
    'num_workers': 8,
    'loss': 'ce',
    'metrics': 'dice'
}
