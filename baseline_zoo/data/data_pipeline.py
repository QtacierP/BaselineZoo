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
        for transform, augs in self.config.data.train_augmentations.items():
            train_transform.append(transforms_list[transform](**augs))
        train_transform += [ToTensor(), Normalize(mean=self.config.data.mean, std=self.config.data.std)]
        for transform, augs in self.config.data.test_augmentations.items():
            test_transform.append(transforms_list[transform](**augs))
        test_transform += [ToTensor(), Normalize(mean=self.config.data.mean, std=self.config.data.std)]
        return Compose(train_transform), Compose(test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.train.batch_size_per_gpu, 
                          num_workers=self.config.train.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.train.batch_size_per_gpu, 
                          num_workers=self.config.train.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.train.batch_size_per_gpu, 
                          num_workers=self.config.train.num_workers)
