__all__ = [
    "SERModule",
    "build_from_yaml",

    "get_arch_by_name",
    "list_archs",

    "get_dataset_by_name",
    "list_datasets",

    "get_loss_by_name",
    "list_losses",

    "get_optimizer_by_name",
    "list_optimizers",

    "get_metric_by_name",
    "list_metrics",

    "get_scheduler_by_name",
    "list_schedulers"
]


from typing import Tuple
import os
import yaml
import pytorch_lightning as pl

from ..module import SERModule
from ..arch import (
    get_arch_by_name,
    list_archs
)

from ..dataset import (
    get_dataset_by_name,
    list_datasets
)

from ..loss import (
    get_loss_by_name,
    list_losses
)

from ..optimizer import (
    get_optimizer_by_name,
    list_optimizers
)

from ..metric import (
    get_metric_by_name,
    list_metrics
)

from ..scheduler import (
    get_scheduler_by_name,
    list_schedulers
)

from .. import utils

def build_from_yaml(yaml_path: str, resume: bool = False) -> Tuple[pl.LightningModule, pl.Trainer]:
    """Builds model and trainer using given yaml file path

    Args:
        yaml_path (str): path of the yaml file
        resume (bool): if true than training will be resumed from checkpoint

    Returns:
        Tuple[pl.LightningModule, pl.Trainer]: model and trainer
    """
    assert os.path.isfile(yaml_path), "given file {} is not exists".format(yaml_path)
    assert yaml_path.endswith(".yaml") or yaml_path.endswith(".yml"), "given file {} must be yaml".format(yaml_path)
    with open(yaml_path, "r") as foo:
        configs = yaml.load(foo, Loader=yaml.FullLoader)

    assert "arch" in configs, "yaml must contain `arch` key"
    assert "hparams" in configs, "yaml must contain `hparams` key"
    assert "trainer" in configs, "yaml must contain `trainer` key"

    arch = configs["arch"]
    hparams = configs["hparams"]
    metrics = configs.get("metrics", {})
    trainer_configs = configs["trainer"]

    if "checkpoint" in trainer_configs:
        checkpoint_configs = trainer_configs.pop("checkpoint")
        checkpoint_configs["filename"] = checkpoint_configs["filename"].format(arch=arch)
        ckpt_file_path = os.path.join(checkpoint_configs["dirpath"], checkpoint_configs["filename"] + ".ckpt")

        trainer_configs["callbacks"] = pl.callbacks.ModelCheckpoint(
            **checkpoint_configs
        )
        if resume:
            trainer_configs.update({"resume_from_checkpoint": ckpt_file_path})

    model = SERModule.build(arch, hparams, metrics)

    trainer = pl.Trainer(**trainer_configs)

    return (model, trainer)