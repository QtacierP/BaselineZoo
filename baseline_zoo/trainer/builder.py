from pytorch_lightning import trainer
from pytorch_lightning.trainer import Trainer
from baseline_zoo.callbacks import callbacks_list
from baseline_zoo.loggers import logger_list
from pytorch_lightning.loggers import LoggerCollection

def build_trainer(config, args):
    callbacks = []
    loggers = []
    keys = {}
    # Build callbacks
    for callback, aug in config.train.callbacks.items():
        callbacks.append(callbacks_list[callback](**aug))
    # Build loggers
    for logger, aug in config.train.log.items():
        loggers.append(logger_list[logger](**aug))
    if len(loggers) > 1:
        logger = LoggerCollection(loggers)
    else:
        logger = loggers[0]
    
    keys['callbacks'] = callbacks
    keys['logger'] = logger

    keys['gpus'] = args.gpu

    if config.train.lr == 'auto':
        keys['auto_lr_find'] = True
    
    keys['benchmark'] = True

    keys['default_root_dir'] = config.model.model_dir
    keys['check_val_every_n_epoch'] = config.train.val_freq 
    keys['max_epochs'] = config.train.epochs 
    return Trainer(**keys) 