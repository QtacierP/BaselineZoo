import pytorch_lightning as pl
from torch.utils.data import DataLoader
from baseline_zoo.data.transforms import transforms_list
from torchvision.transforms import ToTensor, Normalize, Compose

class DataPipeline(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup()
    
    def setup(self):
        self._build_dataset()

    def _build_dataset(self):
        pass # TODO
    
    def _build_transforms(self):
        train_transform = []
        test_transform = []
        for transform, augs in self.config.data.train_augmentations:
            train_transform.append(transforms_list[transform](*augs))
        train_transform += [Normalize(mean=self.config.data.mean, std=self.config.data.std), ToTensor()]
        for transform, augs in self.config.data.test_augmentations:
            test_transform.append(transforms_list[transform](*augs))
        test_transform += [Normalize(mean=self.config.data.mean, std=self.config.data.std), ToTensor()]
        return Compose(train_transform), Compose(test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
