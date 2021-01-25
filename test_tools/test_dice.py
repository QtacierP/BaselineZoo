from pytorch_lightning import metrics
import sys
sys.path.append('/data2/chenpj/BaselineZoo/')
from baseline_zoo.segmentation.metrics.dice import DiceScore
import torch
from pytorch_lightning.metrics.functional import dice_score




if __name__ == '__main__':
    x = torch.ones((2, 2, 128, 128))
    y = torch.zeros((2, 128, 128))
    metric = DiceScore(num_classes=2)
    metric(x, y)
    print(metric.compute_update())
    print(dice_score(x, y))