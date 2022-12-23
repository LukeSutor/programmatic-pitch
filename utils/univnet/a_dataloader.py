import os
import glob
import torch
import random
import numpy as np
import blobfile as bf
import torchaudio
from torch.utils.data import DistributedSampler, DataLoader, Dataset
from collections import Counter

from omegaconf import OmegaConf


from utils import read_wav_np
from stft import TacotronSTFT


def create_dataloader(hp, args, train, device):
    if train:
        dataset = MelFromDisk(hp, hp.data.train_dir, args, train, device)
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=False,
                          num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)

    else:
        dataset = MelFromDisk(hp, hp.data.val_dir, args, train, device)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)


def _list_wav_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["wav"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_wav_files_recursively(full_path))
    return results


class MelFromDisk(Dataset):
    def __init__(self, hp, data_dir, args, train, device):
        super().__init__()
        random.seed(hp.train.seed)
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = data_dir
        self.all_files = _list_wav_files_recursively(data_dir)
        self.stft = TacotronSTFT(hp.audio.filter_length, hp.audio.hop_length, hp.audio.win_length,
                                 hp.audio.n_mel_channels, hp.audio.sampling_rate,
                                 hp.audio.mel_fmin, hp.audio.mel_fmax, center=False, device=device)

        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length



    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if self.train:
            return self.my_getitem(idx)
        else:
            return self.my_getitem(idx)

    def my_getitem(self, idx):
        wavpath = self.all_files[idx]
        # audio, sr = torchaudio.load(wavpath)


        # audio = self._resample_if_necessary(audio, sr)
        # torchaudio.save('test.wav', audio, hp.audio.sampling_rate)
        sr, audio = read_wav_np(wavpath)


        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio).unsqueeze(0)
        mel = self.get_mel(wavpath)

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length -1
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hp.audio.hop_length
            audio_len = self.hp.audio.segment_length
            audio = audio[:, audio_start:audio_start + audio_len]

        return mel, audio

    def get_mel(self, wavpath):
        melpath = wavpath.replace('.wav', '.mel')
        try:
            mel = torch.load(melpath, map_location='cpu')
            assert mel.size(0) == self.hp.audio.n_mel_channels, \
                'Mel dimension mismatch: expected %d, got %d' % \
                (self.hp.audio.n_mel_channels, mel.size(0))

        except (FileNotFoundError, RuntimeError, TypeError, AssertionError):
            sr, wav = read_wav_np(wavpath)
            assert sr == self.hp.audio.sampling_rate, \
                'sample mismatch: expected %d, got %d at %s' % (self.hp.audio.sampling_rate, sr, wavpath)

            if len(wav) < self.hp.audio.segment_length + self.hp.audio.pad_short:
                wav = np.pad(wav, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(wav)), \
                             mode='constant', constant_values=0.0)

            wav = torch.from_numpy(wav).unsqueeze(0)
            mel = self.stft.mel_spectrogram(wav)

            mel = mel.squeeze(0)

            torch.save(mel, melpath)

        return mel

    def _resample_if_necessary(self, signal, sr):
        if sr != self.hp.audio.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.hp.audio.sampling_rate)
            signal = resampler(signal)
        return signal

if __name__ == "__main__":
    hp = OmegaConf.load("./default_c32.yaml")

    dataset = MelFromDisk(hp, hp.data.train_dir, "asdf", True, device="cpu")

    print(dataset.__getitem__(0))