- [ ] log hyperparameters into tensorboard using [this](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams)
- [x] build a yaml structure for hyperparameters
- [ ] integrate yaml with Trainer and LightningModule
- [x] implement `LightningModule.build` function
- [ ] implement `LightningModule.from_checkpoint` function
- [ ] implement `LightningModule.from_pretrained` function (does not include optimizer and scheduler step, and downloads the weights)
- [ ] test saving checkpoint and reloading
- [x] handle model naming
- [x] add profile option for `Trainer` using [this](https://pytorch-lightning.readthedocs.io/en/latest/advanced/profiler.html#performance-and-bottleneck-profiler)
- [ ] Quantization Support
- [ ] Pruning Support
- [ ] yaml tensorboard option


# Basic APIs
- [x] get arch by name
- [x] list archs
- [ ] use given arch if not str

- [x] get dataset by name
- [x] list datasets

- [x] get loss by name
- [x] list losses
- [ ] use given loss if not str

- [x] get optimizer by name
- [x] list optimizers
- [ ] use given optimizer

- [x] get metric by name
- [x] list metrics

- [x] get scheduler by name
- [x] list schedulers
- [ ] use given scheduler

# Utility APIs
- [ ] pick best checkpoint api
- [ ] downsample given dataset