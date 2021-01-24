from configs.eyepacs.eyepacs_base import data_configs

model_configs = {
    'model_dir': './experiment/eyepacs/densenet121',
    'model_name': 'densenet121',
    'n_layers': None,
    'n_downs': None,
    'pre_trained': False,
    'sync_bn': False
}

train_configs = {
    'log_dir': './experimrnt/eyepacs/densenet121/logs/',
    'optimizer': ('adam', {}),
    'scheduler': None,
    'lr': 'auto',
    'epochs': 50,
    'warmup_epochs': 0,
    'save_freq': 10,
    'batch_size_per_gpu': 2,
    'callbacks': {'ModelCheckpoint'  : {'filepath': './experiment/eyepacs/densenet121/models/', 'monitor': 'val_loss'},
                  'ProgressBar'      : {}, 
                  'LearningRateMonitor': {'logging_interval': 'step'}},
    'log': {'TensorBoardLogger': {'save_dir': './experiment/eyepacs/densenet121/logs/'}}, 
    'val_freq': 1,
    'precision': 32,
    'num_workers': 8,
    'loss': 'ce',
    'metrics': 'accuracy'
}

