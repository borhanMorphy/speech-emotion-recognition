from typing import List, Dict
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from src.dataset import EmodbDataset

# TODO select dataset via argparse

def extract_time(audio_files_path: str) -> float:
    signal, sr = librosa.load(audio_files_path)
    return signal.shape[0] / sr

def calc_time(root_path: str) -> Dict[str, List[float]]:
    ds = EmodbDataset(root_path)

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

def main():

    dataset_path = "./data/emodb"
    labels, sizes = zip(*[(k,v) for k,v in get_sizes(dataset_path).items()])
    explode = tuple((0.1 for _ in labels))

    total_time = sum(sizes)

    fig, ax = plt.subplots()
    print("total audio time is {} seconds".format(total_time))
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


if __name__ == '__main__':
    main()