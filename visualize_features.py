import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
from src.dataset import EmodbDataset

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 13

# TODO select dataset via argparse

def get_random_sample_each(root_path:str, k: int = 1):
    ds = EmodbDataset(root_path)
    samples = {}

    for label in ds.get_labels():
        samples[label] = []
        idx = ds.label2id(label)
        ids, = np.where(idx == ds.targets)

        all_selected = random.sample(ids.tolist(), k=k)
        for selected in all_selected:
            samples[label].append(ds.ids[selected])

    return samples

def show_plt(signal: np.ndarray, spec: np.ndarray, MFCCs:np.ndarray, label: str, idx: int):
    fig, axs = plt.subplots(figsize=(6, 8), nrows=3)
    fig.suptitle("{}_{}".format(label, idx))

    # plot raw signal
    axs[0].set_title('raw signal')
    librosa.display.waveplot(signal, sr=SAMPLE_RATE, ax=axs[0])

    # plot log spectrogram
    axs[1].set_title('log spectrogram')
    librosa.display.specshow(spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, ax=axs[1])
    # TODO add colorbar

    # plot mfccs
    axs[2].set_title('MFCCs')
    librosa.display.specshow(MFCCs, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, ax=axs[2])

def main():
    # TODO make dataset path prettier
    dataset_path = "./data/emodb"

    samples = get_random_sample_each(dataset_path, k=1)

    for label,values in samples.items():
        for i, audio_path in enumerate(values):
            signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
            assert sample_rate==SAMPLE_RATE
            
            sfft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
            # sfft: 1 + N_FFT/2, signal_length / HOP_LENGTH
 
            # get real part
            spectrogram = np.abs(sfft)
            # spectrogram: 1 + N_FFT/2, signal_length / HOP_LENGTH
 
            # to log scale
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            # log_pectrogram: 1 + N_FFT/2, signal_length / HOP_LENGTH
 
            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
            # MFCCs: signal_length / HOP_LENGTH, N_MFCC
 
            show_plt(signal, log_spectrogram, MFCCs, label, i)

    plt.show()


if __name__ == '__main__':
    main()