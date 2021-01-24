from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ExponentialLR

scheduler_list = {
    'cosine': CosineAnnealingLR,
    'mutli-step': MultiStepLR,
    'exp': ExponentialLR
}