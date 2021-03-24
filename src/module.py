from typing import Dict, Union, List, Callable
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

from . import api

# TODO pydoc

class SERModule(pl.LightningModule):

    def __init__(self, arch: nn.Module, hparams: Dict = {}, metrics: Dict = {}):
        super().__init__()
        # save hyper parameters
        self.save_hyperparameters(hparams)

        # set architecture
        self.__arch = arch

        # initialize metrics
        self.__metrics = {}
        for metric, values in metrics.items():
            if values is None:
                values = {}
            args = values.get("args", ())
            kwargs = values.get("kwargs", {})
            self.add_metric(
                metric,
                api.get_metric_by_name(metric, *args, **kwargs))
        
        # initialize loss
        self.__loss_fn = api.get_loss_by_name(
            hparams.get("loss", {"id": "CE"}).get("id")
        )

    def add_metric(self, name:str, metric:pl.metrics.Metric):
        self.__metrics[name] = metric

    def set_loss_fn(self, loss: Union[str, nn.Module]):
        assert isinstance(loss, (str, nn.Module)), "loss must be eather string or nn.Module but found {}".format(type(loss))
        if isinstance(loss, str):
            loss = api.get_loss_by_name(loss)
        self.__loss_fn = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__arch.forward(x)

    def prepare_batch(self, raw_audio_signals: Union[np.ndarray, List]) -> torch.Tensor:
        # TODO handle here
        features = librosa.feature.mfcc(raw_audio_signals, n_fft=2048, hop_length=512, n_mfcc=13)
        batch = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)
        return batch.to(self.device, self.dtype)

    @torch.no_grad()
    def predict(self, raw_audio_signals: Union[np.ndarray, List]):
        # TODO implement here
        return 

    def on_train_epoch_start(self):
        # TODO log hyperparameters/lr
        pass

    def training_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch)

        loss = self.__loss_fn(logits, targets)

        return loss

    def on_train_epoch_end(self, outputs):
        losses = [output["minimize"] for output in outputs[0][0]]
        mean_loss = sum(losses) / len(losses)
        self.log("loss/train", mean_loss.item())

    def on_validation_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch)

        preds = F.softmax(logits, dim=1).argmax(dim=1)

        loss = self.__loss_fn(logits, targets)

        for metric in self.__metrics.values():
            metric(preds.cpu(), targets.cpu())

        return loss.item()

    def validation_epoch_end(self, val_outputs: List):
        loss = sum(val_outputs)/len(val_outputs)
        
        for key, metric in self.__metrics.items():
            self.log("metrics/{}".format(key), metric.compute())
        self.log('loss/val', loss)

    def on_test_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch)

        preds = F.softmax(logits, dim=1).argmax(dim=1)

        loss = self.__loss_fn(logits, targets)

        for metric in self.__metrics.values():
            metric(preds.cpu(), targets.cpu())

        return loss.item()

    def on_test_end(self, test_outputs: List):
        loss = sum(test_outputs)/len(test_outputs)
        for key, metric in self.__metrics.items():
            self.log(key, metric.compute())
        self.log('loss/test', loss)

    def configure_optimizers(self):

        # initialize optimizer
        optimizer_cfg = self.hparams.get("optimizer", {"id": "adam", "lr": 1e-3, "weight_decay": 0}).copy()
        optimizer_name = optimizer_cfg.pop("id")
        optimizer_cfg.update({"lr": self.hparams.lr})
        optimizer = api.get_optimizer_by_name(
            self.parameters(), optimizer_name, **optimizer_cfg
        )

        # initialize scheduler
        scheduler_cfg = self.hparams.get("scheduler", {"id": None}).copy()
        scheduler = None
        if ("id" in scheduler_cfg) and (scheduler_cfg["id"] is not None):
            scheduler_name = scheduler_cfg.pop("id")
            scheduler = api.get_scheduler_by_name(
                optimizer, scheduler_name, **scheduler_cfg)

        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    @classmethod
    def build(cls, arch: Union[str, nn.Module], hparams: Dict, metrics: Dict) -> pl.LightningModule:
        assert isinstance(arch, (str, nn.Module)), "architecture must be eather string or nn.Module but found {}".format(type(arch))
        if isinstance(arch, str):
            arch = api.get_arch_by_name(arch, **hparams.get("kwargs", {}))
        return cls(arch, hparams, metrics)

    # TODO from_pretrained()
    # TODO from_checkpoint()

    def on_load_checkpoint(self, checkpoint: Dict):
        print(checkpoint)
        # TODO implement here
        # checkpoint['hyper_parameters']

        # get architecture nn.Module class
        # arch_cls = get_arch_cls(arch)

        # get architecture configuration if needed
        # config = config if isinstance(config,Dict) else get_arch_config(arch,config)

        # build nn.Module with given configuration
        # self.arch  = arch_cls(config=config, **kwargs)
        pass