from baseline_zoo.data import FoldClassificationDataPipeline, \
CifarPipeline
from baseline_zoo.data.foldsegmentation import FoldSegmentationDataPipeline

dataset_list = {
    'fold-classification': FoldClassificationDataPipeline,
    'cifar': CifarPipeline,
    'fold-segmentation': FoldSegmentationDataPipeline
}