from typing import Union, Optional, Dict
import logging
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl
from . import api

# TOOD pydoc

logger = logging.getLogger(__name__)

class SERTrainer:
    def __init__(self, gpus: int = 1,
        accumulate_grad_batches: int = 1, max_epochs: int = 100, 
        gradient_clip_val: float = 0., precision: int = 32,
        profiler: Union[bool, str] = False, auto_lr_find: bool = False,
        check_val_every_n_epoch: int = 1,
        default_root_dir: str = "./", **kwargs):

        self.__trainer = pl.Trainer(gpus=gpus,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=max_epochs, gradient_clip_val=gradient_clip_val,
            precision=precision, profiler=profiler,
            auto_lr_find=auto_lr_find, check_val_every_n_epoch=check_val_every_n_epoch,
            default_root_dir=default_root_dir, **kwargs)

    def init_arch(self, arch: Union[str, nn.Module], num_classes: int = None, **hparams):
        self.__model = api.SERModule.build(arch, num_classes=num_classes, **hparams)

    def fit(self, train_dl, val_dl=None, test_dl=None):
        