# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torchvision.models as models
import torch
from baseline_zoo.classification.model.classfier import BaselineClassifier
import torch.nn.functional as F


class DenseNet(BaselineClassifier):
    def __init__(self, config):
        super().__init__(config)
        backbone = getattr(models, config.model.model_name)(pretrained=config.model.pre_trained)
        num_filters = backbone.classifier.in_features
        self.feature_extractor = backbone.features
        self.classifier = torch.nn.Linear(num_filters, config.data.n_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
