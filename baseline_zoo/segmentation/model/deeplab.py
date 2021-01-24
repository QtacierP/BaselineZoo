import torchvision.models.segmentation as models
from baseline_zoo.segmentation.model.segmentor import BaselineSegmentor
import torch

class DeepLab(BaselineSegmentor):
    def __init__(self, config):
        super().__init__(config)
        self.segment_network = getattr(models, config.model.model_name)(retrained = \
        config.model.pre_trained, progress=True, num_classes=config.data.n_classes)

    def forward(self, x):
        out = self.segment_network(x)
        return out