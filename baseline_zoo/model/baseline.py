import pytorch_lightning as pl
import abc
import torch
from baseline_zoo.optimizer import optimizer_list
from baseline_zoo.metrics import metrics_list
from baseline_zoo.losses import loss_list
from baseline_zoo.scheduler import scheduler_list


class BaselineModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_lr()
        self.configure_loss()
        self.configure_metrics()
    
    def configure_loss(self):
        self.loss = loss_list[self.config.train.loss]

    def configure_metrics(self):
        metrics = metrics_list[self.config.train.metrics]
        self.train_metrics = metrics()
        self.val_metrics = metrics(compute_on_step=False)
    
    def init_lr(self):
        if isinstance(self.config.train.lr, str):
            self.learning_rate = 0.02
        else:
            self.learning_rate = self.config.train.lr

    def configure_optimizers(self):
        optimizer = optimizer_list[self.config.train.optimizer[0]](params=self.parameters(), lr=self.learning_rate, 
                                                              **self.config.train.optimizer[1])
        if self.config.train.scheduler is None:
            return optimizer
        scheduler = scheduler_list[self.config.train.scheduler[0]](optimizer, **self.config.train.scheduler[1])
        return [optimizer], [scheduler]

