from typing import List
import torch.nn as nn
from .naive_cnn.module import NaiveCNN

__arch_mapper__ = {
    "NaiveCNN": NaiveCNN
}

def list_archs() -> List[str]:
    """Returns list of available architecture names

    Returns:
        List[str]: list of architecture names as string

    >>> import src
    >>> src.list_archs()
    ['NaiveCNN']
    """
    return sorted(__arch_mapper__.keys())

def get_arch_by_name(arch: str, *args, **kwargs) -> nn.Module:
    """Returns nn.Module using given `arch`, `args` and `kwargs`

    Args:
        arch (str): name of the architecture

    Returns:
        nn.Module: nn.Module of the architecture

    >>> import src
    >>> model = src.get_arch_by_name("NaiveCNN")
    >>> type(model)
    <class 'src.arch.naive_cnn.module.NaiveCNN'>
    """
    assert arch in __arch_mapper__, "given arch {} is not found".format(arch)
    return __arch_mapper__[arch](*args, **kwargs)