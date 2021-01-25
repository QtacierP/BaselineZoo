# -*- coding: utf-8 -*-
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.nn import parameter
import torch.nn.functional as F
from baseline_zoo.optimizer import optimizer_list
from baseline_zoo.model.baseline import BaselineModel
from baseline_zoo.segmentation.metrics import metrics_list
import pytorch_lightning as pl
import torch


class BaselineSegmentor(BaselineModel):
    def __init__(self, config):
        super().__init__(config)

    def configure_loss(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def configure_metrics(self):
        metrics = metrics_list[self.config.train.metrics]
        augs = {}
        if self.config.train.metrics == 'dice':
            augs['num_classes'] = self.config.data.n_classes
        self.train_metrics = metrics(**augs)
        self.val_metrics = metrics(**augs, compute_on_step=False)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['gt']
        print(torch.unique(y))
        logits = self(x)
        loss = self.loss(logits, y)
        self.train_metrics(logits, y)
        self.log('train_loss', loss)
        self.log_metrics('train')
        self.log("lr", self.learning_rate, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['gt']
        out = self(x)
        val_loss = self.loss(out, y)
        self.val_metrics(out, y)
        self.log('val_loss', val_loss)
        self.log_metrics('val')
    
    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['gt']
        out = self(x)
        val_loss = self.loss(out, y)
        self.val_metrics(out, y)
        self.log('test_loss', val_loss)
        self.log_metrics('test')
    
