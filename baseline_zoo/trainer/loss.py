import torch

loss_list = {
    'ce': torch.nn.CrossEntropyLoss,
    'bce': torch.nn.BCELoss,
    'mse': torch.nn.MSELoss,
    'l1': torch.nn.L1Loss
}