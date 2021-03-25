import numpy as np
import torch
import librosa
import argparse

import src

def parse_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument("--batch-size", "-bs", type=int, default=32)

    ap.add_argument("--yaml-file", "-y", type=str,
        default="./configs/baseline_cnn.yaml", help="yaml file path")

    ap.add_argument("--dataset", "-ds", type=str, choices=src.list_datasets(),
        default="emodb", help="name of the dataset")

    ap.add_argument("--data-splits", "-dsp",
        type=lambda s: [float(i.strip()) for i in s.split(",")], default="0.6, 0.2, 0.2",
        help="train, val, test ratios with given order, splitted with comma")

    ap.add_argument("--n-fft", "-nfft", type=int,
        help="sample size for short time fourier transform", default=2048)

    ap.add_argument("--hop-length", "-hl", type=int, default=512)

    ap.add_argument("--n-mfcc", "-nmfcc", type=int,
        help="number of mel-frequency cepstrum coefficients to be extracted from audio signal", default=13)

    ap.add_argument("--auto-lr", "-alr", action="store_true",
        help="if true than it will try to select learning rate automatically")

    ap.add_argument("--resume", "-r", action="store_true",
        help="if true than training will resume from checkpoint")

    return ap.parse_args()

def collate_fn(batch):
    features, targets = zip(*batch)

    targets = torch.tensor(targets, dtype=torch.int64) # long tensor

    features = np.stack(features, axis=0)
    features = torch.from_numpy(features).float().unsqueeze(1)

    return features, targets

class Transform():
    def __init__(self, n_fft: int, hop_length: int, n_mfcc: int):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(audio, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mfcc=self.n_mfcc)

def main(args):

    model, trainer = src.build_from_yaml(args.yaml_file, resume=args.resume)

    transform = Transform(args.n_fft, args.hop_length, args.n_mfcc)

    ds = src.get_dataset_by_name(args.dataset, transform=transform)

    train_ds, val_ds, test_ds = src.utils.data.random_split(ds, args.data_splits)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4, collate_fn=collate_fn)

    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2, collate_fn=collate_fn)

    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2, collate_fn=collate_fn)

    if (not args.resume) and args.auto_lr:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model,
            train_dataloader=train_dl, val_dataloaders=[val_dl],
            min_lr=1e-5, max_lr=1e-1)

        # Plot with
        lr_finder.plot(suggest=True, show=True)

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print("learning rate suggestion: ", new_lr)
        # update hparams of the model
        model.hparams.lr = new_lr

    # training / validation loop
    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=[val_dl])

    # test loop
    trainer.test(model, test_dataloaders=[test_dl])

if __name__ == '__main__':
    args = parse_arguments()
    main(args)