import os 
from glob import glob
import numpy as np
from PIL import Image
import random 
import torch
from torch.utils.data import Dataset
from torchvision.transforms.transforms import ToTensor
import cv2


class SegmentationDataset(Dataset):
    def __init__(self, root, transform, gt_transform):
        image_dir = os.path.join(root, 'image')
        gt_dir = os.path.join(root, 'gt')
        self.image_list = sorted(glob(os.path.join(image_dir, '*')))
        self.gt_list = sorted(glob(os.path.join(gt_dir,  '*')))
        assert len(self.image_list) == len(self.gt_list)
        self.transform = transform 
        self.gt_transform = gt_transform
    
    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        gt = Image.open(self.gt_list[idx])
        gt = self._pre_process_gt(gt)
        seed = random.randint(0, 999999)
        random.seed(seed)
        torch.random.manual_seed(seed)
        image_tensor = self.transform(image)
        random.seed(seed)
        torch.random.manual_seed(seed)
        gt = self.gt_transform(gt)
        gt_tensor = self._post_process_gt(gt)
        return {'image': image_tensor, 'gt': gt_tensor}

    def _pre_process_gt(self, gt):
        # Deal with different datasets
        return gt

    def _post_process_gt(self, gt):
        gt_tensor = torch.from_numpy(np.array(gt, dtype=np.long))
        # Deal with different datasets
        return (gt_tensor / 255.).long()
     
    def __len__(self) -> int:
        return len(self.image_list)