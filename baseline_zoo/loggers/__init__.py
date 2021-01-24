from pytorch_lightning.loggers import TensorBoardLogger, \
LightningLoggerBase, LoggerCollection, CSVLogger

logger_list = {
    'TensorBoardLogger': TensorBoardLogger,
    'CSVLogger': CSVLogger
}