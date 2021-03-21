from typing import Dict, Union, List
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

class SpeechEmotionRecognizer(pl.LightningModule):

    def __init__(self, arch: nn.Module, hparams: Dict = None):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.__arch = arch
        # TODO make it parametric
        self.__loss_fn = F.cross_entropy

        # initialize empty metrics
        self.__metrics = {}

    def add_metric(self, name:str, metric:pl.metrics.Metric):
        self.__metrics[name] = metric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__arch.forward(x)

    def prepare_batch(self, raw_audio_signals: Union[np.ndarray, List]) -> torch.Tensor:
        # TODO handle here
        features = librosa.feature.mfcc(raw_audio_signals, n_fft=2048, hop_length=512, n_mfcc=13)
        batch = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)
        return batch.to(self.device, self.dtype)

    @torch.no_grad
    def predict(self, raw_audio_signals: Union[np.ndarray, List]):

        batch = self.prepare_batch(raw_audio_signals)

        logits = self.forward(batch)

        scores, preds = logits.max(dim=1)

        return 

    def training_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch.to(self.device, self.dtype))

        loss = self.__loss_fn(logits, targets.to(self.device))

        return loss

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch.to(self.device, self.dtype))

        preds = F.softmax(logits, dim=1).cpu().argmax(dim=1)

        loss = self.__loss_fn(logits, targets.to(self.device))

        return {
            "loss": loss.item(),
            "preds": preds,
            "gts": targets
        }

    @torch.no_grad
    def test_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch.to(self.device, self.dtype))

        preds = F.softmax(logits, dim=1).cpu().argmax(dim=1)

        loss = self.__loss_fn(logits, targets.to(self.device))

        return {
            "loss": loss.item(),
            "preds": preds,
            "gts": targets
        }

    def configure_optimizers(self):
        pass