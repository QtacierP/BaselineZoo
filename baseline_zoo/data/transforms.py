from torchvision import transforms


transforms_list = {
    'resize': transforms.Resize, 
    'horizontal_flip': transforms.RandomHorizontalFlip,
    'vertical_flip': transforms.RandomVerticalFlip,
    'random_resize_crop': transforms.RandomResizedCrop,
    'affine': transforms.RandomAffine,
    'color_jitter': transforms.ColorJitter}