# Module
- [x] implement `LightningModule.build` function
- [ ] implement `LightningModule.from_pretrained` function (does not include optimizer and scheduler step, and downloads the weights)

# Training
- [ ] yaml tensorboard option
- [ ] Quantization Support
- [ ] Pruning Support
- [x] handle model naming
- [ ] add profile option for `Trainer` using [this](https://pytorch-lightning.readthedocs.io/en/latest/advanced/profiler.html#performance-and-bottleneck-profiler)
- [ ] log hyperparameters into tensorboard using [this](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams)
- [x] build a yaml structure for hyperparameters
- [x] integrate yaml with Trainer and LightningModule
- [x] test saving checkpoint and reloading
- [ ] expand dataset options
- [ ] expand arch options (use papers)
- [ ] research SOTA
- [ ] implement SOTA

# Basic APIs
- [ ] add register api

- [x] get arch by name
- [x] list archs
- [ ] support only str arch

- [x] get dataset by name
- [x] list datasets
- [ ] support only str dataset

- [x] get loss by name
- [x] list losses
- [ ] support only str loss

- [x] get optimizer by name
- [x] list optimizers
- [ ] support only str optimize

- [x] get metric by name
- [x] list metrics
- [ ] support only str metric

- [x] get scheduler by name
- [x] list schedulers
- [ ] support only str scheduler

# Other
- [x] handle checkpoint save name with "/"
- [x] add argparser to train.py
- [ ] add auto download functionality for datasets
- [ ] add val.py script
- [ ] add test.py script
- [ ] add demo.py script
- [ ] add sphinx docs