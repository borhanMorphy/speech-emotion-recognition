import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
import argparse

import src

def get_random_sample_each(dataset: str, k: int = 1):
    ds = src.get_dataset_by_name(dataset)
    samples = {}

    for label in ds.get_labels():
        samples[label] = []
        idx = ds.label2id(label)
        ids, = np.where(idx == ds.targets)

        all_selected = random.sample(ids.tolist(), k=k)
        for selected in all_selected:
            samples[label].append(ds.ids[selected])

    return samples

def show_plt(signal: np.ndarray, spec: np.ndarray, MFCCs:np.ndarray,
        label: str, idx: int, sample_rate: int, hop_length: int):
    fig, axs = plt.subplots(figsize=(6, 8), nrows=3)
    fig.suptitle("{}_{}".format(label, idx))

    # plot raw signal
    axs[0].set_title('raw signal')
    librosa.display.waveplot(signal, sr=sample_rate, ax=axs[0])

    # plot log spectrogram
    axs[1].set_title('log spectrogram')
    librosa.display.specshow(spec, sr=sample_rate, hop_length=hop_length, ax=axs[1])
    # TODO add colorbar

    # plot mfccs
    axs[2].set_title('MFCCs')
    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length, ax=axs[2])

def main(dataset: str, sample_rate: int, n_fft: int, hop_length: int, n_mfcc: int):

    samples = get_random_sample_each(dataset, k=1)

    for label,values in samples.items():
        for i, audio_path in enumerate(values):
            signal, sr = librosa.load(audio_path, sr=sample_rate)
            assert sr==sample_rate
            
            sfft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
            # sfft: 1 + n_fft/2, signal_length / hop_length
 
            # get real part
            spectrogram = np.abs(sfft)
            # spectrogram: 1 + n_fft/2, signal_length / hop_length
 
            # to log scale
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            # log_pectrogram: 1 + n_fft/2, signal_length / hop_length
 
            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
            # MFCCs: signal_length / hop_length, n_mfcc
 
            show_plt(signal, log_spectrogram, MFCCs, label, i, sample_rate, hop_length)

    plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", "-ds", type=str, required=True, choices=src.list_datasets())

    ap.add_argument("--sample-rate", "-sr", type=int, help="sample rate for audio", default=22050)

    ap.add_argument("--n-fft", "-nfft", type=int,
        help="sample size for short time fourier transform", default=2048)

    ap.add_argument("--hop-length", "-hl", type=int, default=512)

    ap.add_argument("--n-mfcc", "-nmfcc", type=int,
        help="number of mel-frequency cepstrum coefficients to be extracted from audio signal", default=13)

    args = ap.parse_args()

    main(args.dataset, args.sample_rate, args.n_fft,
        args.hop_length, args.n_mfcc)