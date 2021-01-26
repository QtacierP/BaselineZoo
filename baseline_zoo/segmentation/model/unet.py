import torchvision.models.segmentation as models
from baseline_zoo.segmentation.model.segmentor import BaselineSegmentor
from baseline_zoo.segmentation.model.backbone import UNetBackbone
import torch

class UNet(BaselineSegmentor):
    def __init__(self, config):
        super().__init__(config)
        self.segment_network = UNetBackbone(n_channels=self.config.data.n_colors, n_classes=self.config.data.n_classes)

    def forward(self, x):
        out = self.segment_network(x)
        return out