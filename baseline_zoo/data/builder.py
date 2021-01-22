from torchvision import transforms
import torchvision.datasets as datasets
from baseline_zoo.data.transforms import transforms_list
import pytorch_lightning as pl




def build_data(config):
    if config.data.data_dir is None:
        print('[INFO]: Data directory is not set in config, finding in torchvision...')
        if 'cifar' in config.data.dataset:
            from baseline_zoo.data.cifar import CifarPipeline
            return CifarPipeline(config)
        else:
            raise NotImplementedError('{} dataset is not supported now'.format(config.data.dataset))
    else:
        if 'cifar' in config.data.dataset:
            from baseline_zoo.data.cifar import CifarPipeline
            return CifarPipeline(config)
        else:
            raise NotImplementedError('{} dataset is not supported now'.format(config.data.dataset))
    