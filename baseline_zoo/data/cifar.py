from torch.utils import data
from torchvision.transforms import transforms
from baseline_zoo.data.data_pipeline import DataPipeline
import torchvision.datasets.cifar as cifar
from torch.utils.data import random_split, DataLoader


class CifarPipeline(DataPipeline):
    def __init__(self, config):
        super().__init__(config)

    def _build_dataset(self):
        try:
            dataset_module = getattr(cifar, self.config.data.dataset.upper())
        except:
            raise NotImplementedError('{} is not supported in Cifar Dataset'.format(self.config.data.dataset))
        train_transform, test_transform = self._build_transforms()
        train_dataset = dataset_module(root=self.config.data.data_dir, train=True, transform=train_transform, download=True)
        self.train_dataset, self.val_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.7), 
            len(train_dataset) - int(len(train_dataset) * 0.7)])
        self.test_dataset = dataset_module(root=self.config.data.data_dir, train=False, transform=test_transform, download=True)


        