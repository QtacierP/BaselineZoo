
import os
from torchvision.transforms import ToTensor, Normalize, Compose
from baseline_zoo.data.data_pipeline import DataPipeline
from baseline_zoo.data.utils import SegmentationDataset
from baseline_zoo.data.transforms import transforms_list


class FoldSegmentationDataPipeline(DataPipeline):
    def __init__(self, config):
        super().__init__(config)
    
    def _build_dataset(self):
        train_transform, test_transform, \
        train_gt_transform, test_gt_transform = self._build_transforms()
        self.train_dataset = SegmentationDataset(os.path.join(self.config.data.data_dir, 'train'),
                                                 transform=train_transform, gt_transform=train_gt_transform)
        self.test_dataset = SegmentationDataset(os.path.join(self.config.data.data_dir, 'test'),
                                                 transform=test_transform, gt_transform=test_gt_transform)
        self.val_dataset = SegmentationDataset(os.path.join(self.config.data.data_dir, 'val'),
                                                 transform=test_transform, gt_transform=test_gt_transform)
    

    def _build_transforms(self):
        train_transform = []
        test_transform = []
        train_gt_transform = []
        test_gt_transform = []
        for transform, augs in self.config.data.train_augmentations.items():
            train_transform.append(transforms_list[transform](**augs))
            train_gt_transform.append(transforms_list[transform](**augs))
        train_transform += [ToTensor(), Normalize(mean=self.config.data.mean, std=self.config.data.std)]
        for transform, augs in self.config.data.test_augmentations.items():
            test_transform.append(transforms_list[transform](**augs))
            test_gt_transform.append(transforms_list[transform](**augs))
        test_transform += [ToTensor(), Normalize(mean=self.config.data.mean, std=self.config.data.std)]
        return Compose(train_transform), Compose(test_transform), \
                Compose(train_gt_transform), Compose(test_gt_transform)

