## Imports
import torchmetrics
import config

## Config variables
NUM_LABELS = config.NUM_LABELS
THRESHOLD = config.THRESHOLD


micro_metric_kwargs = {
    "num_labels": NUM_LABELS,
    "threshold": THRESHOLD,
    "average": "micro",
}
macro_metric_kwargs = micro_metric_kwargs
macro_metric_kwargs["average"] = "macro"

micro_metrics = torchmetrics.MetricCollection(
    [
        torchmetrics.classification.MultilabelPrecision(**micro_metric_kwargs),
        torchmetrics.classification.MultilabelRecall(**micro_metric_kwargs),
        torchmetrics.classification.MultilabelF1Score(**micro_metric_kwargs),
    ]
)

macro_metrics = torchmetrics.MetricCollection(
    [
        torchmetrics.classificationn.MultilabelPrecision(**macro_metric_kwargs),
        torchmetrics.classification.MultilabelRecall(**macro_metric_kwargs),
        torchmetrics.classification.MultilabelF1Score(**macro_metric_kwargs),
    ]
)
