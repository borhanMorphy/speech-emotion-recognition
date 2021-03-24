from typing import List
from torch.utils.data import Dataset

from .emodb_dataset import EmodbDataset

__dataset_mapper__ = {
    "emodb": EmodbDataset
}

def list_datasets() -> List[str]:
    """Returns list of available datasets names

    Returns:
        List[str]: list of dataset names as string

    >>> import src
    >>> src.list_datasets()
    ['emodb']
    """
    return sorted(__dataset_mapper__.keys())

def get_dataset_by_name(dataset: str, *args, **kwargs) -> Dataset:
    """Returns Dataset using given `dataset`, `args` and `kwargs`

    Args:
        dataset (str): name of the dataset

    Returns:
        Dataset: requested dataset as Dataset

    >>> import src
    >>> dataset = src.get_dataset_by_name("emodb")
    >>> type(dataset)
    <class 'src.dataset.emodb_dataset.EmodbDataset'>
    """
    assert dataset in __dataset_mapper__, "given dataset {} is not found".format(dataset)
    return __dataset_mapper__[dataset](*args, **kwargs)
