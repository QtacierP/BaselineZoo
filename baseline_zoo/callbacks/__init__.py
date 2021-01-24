from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import Callback, EarlyStopping, \
GPUStatsMonitor, GradientAccumulationScheduler, LearningRateMonitor, \
ModelCheckpoint, ProgressBar

callbacks_list = {
    'EarlyStopping': EarlyStopping, 
    'GPUStatsMonitor': GPUStatsMonitor,
    'GradientAccumulationScheduler': GradientAccumulationScheduler,
    'LearningRateMonitor': LearningRateMonitor,
    'ModelCheckpoint': ModelCheckpoint,
    'ProgressBar': ProgressBar}