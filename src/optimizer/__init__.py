from typing import List
import torch.optim as optim

__optimizer_mapper__ = {
    "adam": optim.Adam,
    "sgd": optim.SGD
}

def list_optimizers() -> List[str]:
    """Returns list of available optimizers names

    Returns:
        List[str]: list of optimizer names as string

    >>> import src
    >>> src.list_optimizers()
    ['adam','sgd']
    """
    return sorted(__optimizer_mapper__.keys())

def get_optimizer_by_name(parameters, optimizer: str, *args, **kwargs) -> optim.Optimizer:
    """Returns optimizer using given `optimizer`, `args` and `kwargs`

    Args:
        parameters : model weights
        optimizer (str): name of the optimizer

    Returns:
        optim.Optimizer: requested optimizer as `optim.Optimizer`
    """
    assert optimizer in __optimizer_mapper__, "given optimizer {} is not found".format(optimizer)
    return __optimizer_mapper__[optimizer](parameters, *args, **kwargs)
