import os
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
import librosa

SAMPLE_RATE = 22050
DURATION = 1.4 # second

class EmodbDataset(Dataset):
    # url: http://www.emodb.bilderbar.info/download/
    # 16 kHz, 16 bit, mono.
    # speaker language: German
    # angry => {Wa, Wb, Wc, Wd}
    # happy => {Fa, Fb, Fc, Fd}
    # neutral => {Na, Nb, Nc, Nd}
    # sad => {Ta, Tb, Tc, Td}

    __url__ = "http://www.emodb.bilderbar.info/download/download.zip"
    __labels__ = ("angry", "happy", "neutral", "sad")
    __suffixes__ = {
        "angry": ["Wa", "Wb", "Wc", "Wd"],
        "happy": ["Fa", "Fb", "Fc", "Fd"],
        "neutral": ["Na", "Nb", "Nc", "Nd"],
        "sad": ["Ta", "Tb", "Tc", "Td"]
    }

    def __init__(self, root_path: str, transform=None):
        super().__init__()
        assert os.path.isdir(root_path), "given root path: {} must be directory".format(root_path)
        audio_root_path = os.path.join(root_path, "wav")

        ids = []
        targets = []
        for audio_file in os.listdir(audio_root_path):
            f_name, ext = os.path.splitext(audio_file)
            if ext != ".wav": continue

            suffix = f_name[-2:]
            for label, suffixes in self.__suffixes__.items():
                if suffix in suffixes:
                    ids.append(os.path.join(audio_root_path, audio_file))
                    targets.append(self.label2id(label))
                    break

        # TODO resample by lenght

        self.ids = ids
        self.targets = np.array(targets, dtype=np.int64)

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx:int) -> Tuple:
        target = self.targets[idx]
        audio = self.load_audio(self.ids[idx])

        if self.transform:
            audio = self.transform(audio)

        return audio, target

    @staticmethod
    def id2label(idx: int) -> str:
        return EmodbDataset.__labels__[idx]

    @staticmethod
    def label2id(label: str) -> int:
        return EmodbDataset.__labels__.index(label)

    @staticmethod
    def load_audio(audio_file_path: str) -> np.ndarray:
        audio, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE, duration=DURATION)
        assert SAMPLE_RATE == sr, "broken audio file"
        return audio

    @staticmethod
    def get_labels() -> List[str]:
        return list(EmodbDataset.__labels__)
