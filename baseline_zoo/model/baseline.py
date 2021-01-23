import pytorch_lightning as pl
import abc
import torch
from baseline_zoo.trainer.optimizer import optimizer_list
from baseline_zoo.trainer.metrics import metrics_list
from baseline_zoo.trainer.loss import loss_list
from baseline_zoo.trainer.scheduler import scheduler_list


class BaselineModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.configure_loss()
        self.configure_metrics()
    
    def configure_loss(self):
        self.loss = loss_list[self.config.train.loss]

    def configure_metrics(self):
        metrics = metrics_list[self.config.train.metrics]
        self.train_metrics = metrics()
        self.val_metrics = metrics(compute_on_step=False)
        
   
    def configure_optimizers(self):
        if isinstance(self.config.train.lr, str):
            default_lr = 0.02
        else:
            default_lr = self.config.train.lr
        optimizer = optimizer_list[self.config.train.optimizer[0]](params=self.parameters(), lr=default_lr, 
                                                              **self.config.train.optimizer[1])
        if self.config.train.scheduler is None:
            return optimizer
        scheduler = scheduler_list[self.config.train.scheduler[0]](optimizer, **self.config.train.scheduler[1])
        return [optimizer], [scheduler]

