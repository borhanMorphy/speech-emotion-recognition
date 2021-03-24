import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import librosa
from tqdm import tqdm
import argparse

import src

def extract_time(audio_files_path: str) -> float:
    signal, sr = librosa.load(audio_files_path)
    return signal.shape[0] / sr

def calc_time(dataset: str) -> Dict[str, List[float]]:
    ds = src.get_dataset_by_name(dataset)

    vals = {label: [] for label in ds.get_labels()}

    for audio_file_path, target in tqdm(zip(ds.ids, ds.targets), total=len(ds)):
        label = ds.id2label(target)
        vals[label].append( extract_time(audio_file_path) )

    return vals

def main(dataset: str):

    vals = calc_time(dataset)

    labels = list(vals.keys())

    mins = []
    means = []
    maxes = []

    for label in labels:
        min_val = min(vals[label])
        mean_val = sum(vals[label]) / len(vals[label])
        max_val = max(vals[label])
        mins.append( round(min_val, 2) )
        means.append( round(mean_val, 2) )
        maxes.append( round(max_val, 2) )

    x = np.arange(len(labels))  # the label locations
    bar_width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - bar_width, mins, bar_width, label='min')
    rects2 = ax.bar(x, means, bar_width, label='mean')
    rects3 = ax.bar(x + bar_width, maxes, bar_width, label='max')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Seconds')
    ax.set_title('Min & Mean & Max  seconds for each emotion')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "-ds", type=str, required=True, choices=src.list_datasets())
    args = ap.parse_args()
    main(args.dataset)