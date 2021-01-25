from baseline_zoo.data.data_pipeline import DataPipeline 
from torchvision.datasets import ImageFolder
import os

class FoldClassificationDataPipeline(DataPipeline):
    def __init__(self, config):
        super().__init__(config)
    
    def _build_dataset(self):
        train_transform, test_transform = self._build_transforms()
        self.train_dataset = ImageFolder(os.path.join(self.config.data.data_dir, 'train'), transform=train_transform)
        self.test_dataset = ImageFolder(os.path.join(self.config.data.data_dir, 'test'), transform=test_transform)
        self.val_dataset = ImageFolder(os.path.join(self.config.data.data_dir, 'val'), transform=test_transform)
