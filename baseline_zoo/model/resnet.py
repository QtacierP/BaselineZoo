import pytorch_lightning as pl
import torchvision.models as models
import torch

class ResNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        backbone = getattr(models, config.model.model_name)(pretrained=False)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(num_filters, config.data.n_classes)

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x
