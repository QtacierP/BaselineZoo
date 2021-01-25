from configs.drive.drive_base import data_configs


model_configs = {
    'model_dir': './experiment/drive/deeplabv3_resnet50',
    'model_name': 'deeplabv3_resnet50',
    'n_layers': None,
    'n_downs': None,
    'pre_trained': False,
    'sync_bn': False
}

train_configs = {
    'log_dir': './experimrnt/drive/deeplabv3_resnet50/logs/',
    'optimizer': ('adam', {}),
    'scheduler': None,
    'lr': 1e-4,
    'epochs': 50,
    'warmup_epochs': 0,
    'save_freq': 10,
    'batch_size_per_gpu': 2,
    'callbacks': {'ModelCheckpoint'  : {'filepath': './experiment/drive/deeplabv3_resnet50/models/', 'monitor': 'val_loss'},
                  'ProgressBar'      : {}},
    'log': {'TensorBoardLogger': {'save_dir': './experiment/drive/deeplabv3_resnet50/logs/'}}, 
    'val_freq': 1,
    'precision': 32,
    'num_workers': 8,
    'loss': 'ce',
    'metrics': 'accuracy'
}
