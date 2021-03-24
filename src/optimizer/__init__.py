from typing import List
from torch.optim import Optimizer

__optimizer_mapper__ = {}

def list_optimizers() -> List[str]:
    """Returns list of available optimizers names

    Returns:
        List[str]: list of optimizer names as string

    >>> import src
    >>> src.list_optimizers()
    []
    """
    return sorted(__optimizer_mapper__.keys())

def get_optimizer_by_name(optimizer: str, *args, **kwargs) -> Optimizer:
    """Returns optimizer using given `optimizer`, `args` and `kwargs`

    Args:
        optimizer (str): name of the optimizer

    Returns:
        Optimizer: requested optimizer as Optimizer
    """
    assert optimizer in __optimizer_mapper__, "given optimizer {} is not found".format(optimizer)
    return __optimizer_mapper__[optimizer](*args, **kwargs)
