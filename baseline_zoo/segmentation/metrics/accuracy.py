import pytorch_lightning as pl

# TODO

class SegmentationAccuracy(pl.metrics.Accuracy):
    def __init__(self, threshold: float, compute_on_step: bool, dist_sync_on_step: bool, process_group: Optional[Any]):
        super().__init__(threshold=threshold, compute_on_step=compute_on_step, 
        dist_sync_on_step=dist_sync_on_step, process_group=process_group)
    