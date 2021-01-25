import pytorch_lightning as pl
from baseline_zoo.metrics.accuracy import UpdatedAccuracy

metrics_list = {
    'accuracy':  UpdatedAccuracy,
    'mse':  pl.metrics.MeanSquaredError,
    'f1': pl.metrics.F1
}