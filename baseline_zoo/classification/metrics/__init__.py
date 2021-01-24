import pytorch_lightning as pl

metrics_list = {
    'accuracy': pl.metrics.Accuracy,
    'mse':  pl.metrics.MeanSquaredError,
    'f1': pl.metrics.Fbeta
}