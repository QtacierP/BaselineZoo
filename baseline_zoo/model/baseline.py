from logging import warn
import pytorch_lightning as pl
import abc
import torch
from baseline_zoo.optimizer import optimizer_list
from baseline_zoo.metrics import metrics_list
from baseline_zoo.losses import loss_list
from baseline_zoo.scheduler import scheduler_list


class BaselineModel(pl.LightningModule):
    def __init__(self, config):
        """
        The BaselineModel is the basic abstract model of the BaselineZoo project.
        Args:
            config (Config): The core config file
        """
        super().__init__()
        self.config = config
        self.init_lr()
        self.configure_loss()
        self.configure_metrics()
        self.configure_loggers()
    
    def configure_loss(self):
        """
        Build loss function
        """
        self.loss = loss_list[self.config.train.loss]

    def configure_loggers(self):
        """
        Build loggers for logging metrics, loss, meta-data (TODO)
        """
        self.logger_utils = {}
        if hasattr(self, 'train_metrics') and hasattr(self, 'val_metrics'):
            try:
                self.logger_utils['train_metrics'] = self.train_metrics.compute_update
                self.logger_utils['val_metrics'] = self.val_metrics.compute_update
            except:
                warn('{} is in-built, using pytorch-lightning metrics'.format(self.config.train.metrics))
                self.logger_utils['train_metrics'] = self.train_metrics.compute
                self.logger_utils['val_metrics'] = self.val_metrics.compute

    def configure_metrics(self):
        """
        Build metrics 
        """
        metrics = metrics_list[self.config.train.metrics]
        self.train_metrics = metrics()
        self.val_metrics = metrics(compute_on_step=False)
    
    def log_metrics(self, mode='train'):
        """
        Log metrics in traning.
        The default setting is to average the metrics based on each step in one epoch
        Args:
            mode (str, optional): The training stage. Defaults to 'train'.
        """
        if mode == 'train':
            augs = {
                'value': self.logger_utils['train_metrics'](), 
                'on_step':True, 
                'on_epoch':True, 
                'prog_bar':True}
        else:
            augs = {
                'value': self.logger_utils['val_metrics'](), 
                'on_step':False, 
                'on_epoch':True, 
                'prog_bar':True}
        self.log('{}_{}'.format(mode, self.config.train.metrics), **augs)
    
    def init_lr(self):
        """
        Initialize learning rate, which is designed for auto-lr-finder module
        """
        if isinstance(self.config.train.lr, str):
            self.learning_rate = 0.02
        else:
            self.learning_rate = self.config.train.lr

    def configure_optimizers(self):
        """
        Build optimizers and schedulers.
        Returns:
            optimzers or list [optimizers, schedulers]: An internal implementation function in Pytorch-Lightning
        """
        optimizer = optimizer_list[self.config.train.optimizer[0]](params=self.parameters(), lr=self.learning_rate, 
                                                              **self.config.train.optimizer[1])
        if self.config.train.scheduler is None:
            return optimizer
        scheduler = scheduler_list[self.config.train.scheduler[0]](optimizer, **self.config.train.scheduler[1])
        return [optimizer], [scheduler]

