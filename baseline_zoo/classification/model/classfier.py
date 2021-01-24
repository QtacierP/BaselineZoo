import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.nn import parameter
import torch.nn.functional as F
from baseline_zoo.optimizer import optimizer_list
from baseline_zoo.model.baseline import BaselineModel
import pytorch_lightning as pl
import torch


class BaselineClassifier(BaselineModel):
    def __init__(self, config):
        super().__init__(config)

    def configure_loss(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.train_metrics(logits, y)
        self.log('train_loss', loss)  
        self.log('train_acc', self.train_metrics.correct.float() / self.train_metrics.total, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        val_loss = self.loss(out, y)
        self.val_metrics(out, y)
        self.log('val_loss', val_loss)
        self.log('val_acc', self.val_metrics.compute(), on_step=False, on_epoch=True, prog_bar=True)
    
