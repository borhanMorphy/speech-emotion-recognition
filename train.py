import numpy as np
import torch
import librosa

import src

N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 13

PARTS = [0.6, 0.2, 0.2]

BATCH_SIZE = 32

AUTO_LR_FINDER = False


def collate_fn(batch):
    features, targets = zip(*batch)

    targets = torch.tensor(targets, dtype=torch.int64) # long tensor

    features = np.stack(features, axis=0)
    features = torch.from_numpy(features).float().unsqueeze(1)
    return features, targets

class Transform():

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        after= librosa.feature.mfcc(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
        return after

model, trainer = src.build_from_yaml("configs/baseline_cnn.yaml")

ds = src.get_dataset_by_name("emodb", transform=Transform())

train_ds, val_ds, test_ds = src.utils.data.random_split(ds, PARTS)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, collate_fn=collate_fn)

val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, collate_fn=collate_fn)

test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, collate_fn=collate_fn)

if AUTO_LR_FINDER:
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model,
        train_dataloader=train_dl, val_dataloaders=[val_dl],
        min_lr=1e-5, max_lr=1e-1)
    # Plot with
    fig = lr_finder.plot(suggest=True, show=True)

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print("suggestion: ",new_lr)
    # update hparams of the model
    model.hparams.lr = new_lr

trainer.fit(model, train_dataloader=train_dl, val_dataloaders=[val_dl])

trainer.test(model, test_dataloaders=[test_dl])