from typing import List, Dict
import matplotlib.pyplot as plt
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

def get_sizes(root_path: str) -> Dict[str, float]:
    vals = calc_time(root_path)

    for k,v in vals.items():
        vals[k] = sum(v)

    return vals

def main(dataset: str):

    labels, sizes = zip(*[(k,v) for k,v in get_sizes(dataset).items()])
    explode = tuple((0.1 for _ in labels))

    total_time = sum(sizes)

    fig, ax = plt.subplots()
    print("total audio time is {} seconds".format(total_time))
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "-ds", type=str, required=True, choices=src.list_datasets())
    args = ap.parse_args()
    main(args.dataset)