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

    def __init__(self, arch: nn.Module, hparams: Dict = None):
        super().__init__()
        self.save_hyperparameters(hparams)

        # set architecture
        self.__arch = arch

        # initialize loss
        self.__loss_fn = api.get_loss_by_name(
            hparams.get("loss", {"id": "CE"}).get("id")
        )

        # initialize metrics
        self.__metrics = {}
        for metric in hparams.get("metrics", []):
            self.add_metric(metric, api.get_metric_by_name(metric))

    def add_metric(self, name:str, metric:pl.metrics.Metric):
        self.__metrics[name] = metric

    def set_loss_fn(self, loss: Union[str, nn.Module]):
        assert isinstance(loss, (str, nn.Module)), "loss must be eather string or nn.Module but found {}".format(type(loss))
        if isinstance(loss, str):
            loss = api.get_loss_by_name(loss)
        self.__loss_fn = loss

    # TODO def set_optimizer(self) hint: use getter and setter
    # TODO def set_scheduler(self) hint: use getter and setter

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
        batch = self.prepare_batch(raw_audio_signals)

        logits = self.forward(batch)

        scores, preds = logits.max(dim=1)

        return 

    def training_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch.to(self.device, self.dtype))

        loss = self.__loss_fn(logits, targets.to(self.device))

        return loss

    def on_validation_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch.to(self.device, self.dtype))

        preds = F.softmax(logits, dim=1).cpu().argmax(dim=1)

        loss = self.__loss_fn(logits, targets.to(self.device))

        for metric in self.__metrics.values():
            metric(preds, targets)

        return loss.item()

    def validation_epoch_end(self, val_outputs: List):
        loss = sum(val_outputs)/len(val_outputs)
        for key, metric in self.__metrics.items():
            self.log(key, metric.compute())
        self.log('val_loss', loss)

    def on_test_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch.to(self.device, self.dtype))

        preds = F.softmax(logits, dim=1).cpu().argmax(dim=1)

        loss = self.__loss_fn(logits, targets.to(self.device))

        for metric in self.__metrics.values():
            metric(preds, targets)

        return loss.item()

    def on_test_end(self, test_outputs: List):
        loss = sum(test_outputs)/len(test_outputs)
        for key, metric in self.__metrics.items():
            self.log(key, metric.compute())
        self.log('test_loss', loss)

    def configure_optimizers(self):
        pass

    @classmethod
    def build(cls, arch: Union[str, nn.Module], **hparams) -> pl.LightningModule:
        assert isinstance(arch, (str, nn.Module)), "architecture must be eather string or nn.Module but found {}".format(type(arch))
        if isinstance(arch, str):
            arch = api.get_arch_by_name(arch, **hparams.get("kwargs", {}))
        return cls(arch, hparams)

    # TODO from_pretrained()
    # TODO from_checkpoint()

    def on_load_checkpoint(self, checkpoint: Dict):
        print(checkpoint)
        # checkpoint['hyper_parameters']

        # get architecture nn.Module class
        # arch_cls = get_arch_cls(arch)

        # get architecture configuration if needed
        # config = config if isinstance(config,Dict) else get_arch_config(arch,config)

        # build nn.Module with given configuration
        # self.arch  = arch_cls(config=config, **kwargs)
        pass