from typing import List
import torch.nn as nn

__loss_mapper__ = {
    "CE": nn.CrossEntropyLoss
}

def list_losses() -> List[str]:
    """Returns list of available losss names

    Returns:
        List[str]: list of loss names as string

    >>> import src
    >>> src.list_losses()
    ['CE']
    """
    return sorted(__loss_mapper__.keys())

def get_loss_by_name(loss: str, *args, **kwargs) -> nn.Module:
    """Returns loss using given `loss`, `args` and `kwargs`

    Args:
        loss (str): name of the loss

    Returns:
        nn.Module: requested loss as nn.Module
    """
    assert loss in __loss_mapper__, "given loss {} is not found".format(loss)
    return __loss_mapper__[loss](*args, **kwargs)
