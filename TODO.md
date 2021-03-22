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
- [ ] get arch by name
- [ ] list archs
- [ ] use given arch if not str

- [ ] get dataset by name
- [ ] list datasets

- [ ] get loss by name
- [ ] list losses
- [ ] use given loss if not str

- [ ] get optimizer by name
- [ ] list optimizers
- [ ] use given optimizer

- [ ] get metric by name
- [ ] list metrics

- [ ] get scheduler by name
- [ ] list schedulers
- [ ] use given scheduler

# Utility APIs
- [ ] pick best checkpoint api
- [ ] downsample given dataset