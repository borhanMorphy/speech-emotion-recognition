from typing import List
from torch.optim import Optimizer

__scheduler_mapper__ = {}

def list_schedulers() -> List[str]:
    """Returns list of available schedulers names

    Returns:
        List[str]: list of scheduler names as string

    >>> import src
    >>> src.list_schedulers()
    []
    """
    return sorted(__scheduler_mapper__.keys())

def get_scheduler_by_name(optimizer: Optimizer, scheduler: str, *args, **kwargs):
    """Returns scheduler using given `scheduler`, `args` and `kwargs`

    Args:
        optimizer (Optimizer): model optimizer
        scheduler (str): name of the scheduler

    """
    assert scheduler in __scheduler_mapper__, "given scheduler {} is not found".format(scheduler)
    return __scheduler_mapper__[scheduler](optimizer, *args, **kwargs)
