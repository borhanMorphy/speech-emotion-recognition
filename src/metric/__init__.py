from typing import List
from pytorch_lightning.metrics import Metric

__metric_mapper__ = {}

def list_metrics() -> List[str]:
    """Returns list of available metrics names

    Returns:
        List[str]: list of metric names as string

    >>> import src
    >>> src.list_metrics()
    []
    """
    return sorted(__metric_mapper__.keys())

def get_metric_by_name(metric: str, *args, **kwargs) -> Metric:
    """Returns metric using given `metric`, `args` and `kwargs`

    Args:
        metric (str): name of the metric

    Returns:
        Metric: requested metric as Metric
    """
    assert metric in __metric_mapper__, "given metric {} is not found".format(metric)
    return __metric_mapper__[metric](*args, **kwargs)
