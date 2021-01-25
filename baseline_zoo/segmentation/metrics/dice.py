from pytorch_lightning.metrics import Metric
from typing import Any, Callable, Optional, Union
from pytorch_lightning.metrics.functional import dice_score 
from pytorch_lightning.metrics.utils import _input_format_classification
import torch

class DiceScore(Metric):
    # TODO: Only works for samplewise_mean
    def __init__(
        self,
        num_classes: int = 2,
        threshold = 0.5,
        bg: bool = False,
        nan_score: float = 0.0,
        no_fg_score: float = 0.0,
        reduction: str = 'samplewise_mean',
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.threshold = threshold
        self.num_classes = num_classes
        self.bg = bg
        self.bg_num =  (1 - int(bool(bg)))
        self.nan_score = nan_score
        self.no_fg_score = no_fg_score
        self.reduction = reduction
        self.dice_reduction = reduction
        if reduction == 'samplewise_mean':
            self.dice_reduction = 'elementwise_mean'
        
        self.add_state("dice", default=torch.zeros(self.num_classes - self.bg_num, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape
        N_samples = preds.shape[0]
        self.total += N_samples
        for i in range(N_samples):
            self.dice += dice_score(
                            pred=preds[i],
                            target=target[i],
                            bg=self.bg, 
                            nan_score=self.nan_score,
                            no_fg_score=self.no_fg_score,
                            reduction=self.dice_reduction)
            


    def compute(self):
        """
        Computes dice over state.
        """
        return self.dice / self.total
    
    def compute_update(self):
        """
        Computes average dice on each step in one epoch.
        """
        return self.dice / self.total