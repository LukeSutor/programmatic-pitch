import sys
import math
import random

import blobfile as bf
# from mpi4py import MPI
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
import soundfile as sf
#TESTING
sys.path.insert(0, "../../")
# from utils import read_wav_np
import constants

# from stft import TacotronSTFT


def create_dataloader(train, device):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=constants.SAMPLE_RATE,
        n_fft=constants.FILTER_LENGTH,
        hop_length=constants.HOP_LENGTH,
        n_mels=constants.NUM_CHANNELS,
        win_length=constants.WIN_LENGTH,
        f_min=constants.FMIN,
        f_max=constants.FMAX
    )

    if train:
        dataset = AudioDataset(_list_wav_files_recursively(constants.TRAIN_DATA), constants.SAMPLE_RATE, constants.TARGET_SAMPLES, mel_spectrogram, device="cuda")
        return DataLoader(dataset=dataset, batch_size=constants.BATCH_SIZE, shuffle=True,
                          num_workers=constants.NUM_WORKERS, pin_memory=True, drop_last=True)
    else:
        dataset = AudioDataset(_list_wav_files_recursively(constants.VALID_DATA), constants.SAMPLE_RATE, constants.TARGET_SAMPLES, mel_spectrogram, device="cuda")
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=constants.NUM_WORKERS, pin_memory=True, drop_last=False)


def load_info(path: str) -> dict:
    """Load audio metadata
    this is a backend_independent wrapper around torchaudio.info
    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds
    """
    # get length of file in samples
    if torchaudio.get_audio_backend() == "sox":
        raise RuntimeError("Deprecated backend is not supported")

    info = {}
    si = torchaudio.info(str(path))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info

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


class AudioDataset(Dataset):
    def __init__(
        self,
        image_paths,
        target_sample_rate,
        target_samples,
        transformation,
        device="cpu"
    ):
        super().__init__()
        self.local_images = image_paths
        self.device = device
        self.transformation = transformation#.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.target_samples = target_samples
        

    def __len__(self):
        return len(self.local_images)


    def __getitem__(self, idx):
        path = self.local_images[idx]

        # Get a random sequence from the data of length sr * audio_length
        audio_data = load_info(path)
        duration = audio_data["samples"]
        sample_length = math.floor(256 * 1023 * audio_data["samplerate"] / self.target_sample_rate)
        start = random.randint(0, duration - sample_length)

        signal, sr = torchaudio.load(path, num_frames = sample_length, frame_offset = start)
        # signal = signal.to(self.device)

        # Resample to make all sample rates the same
        signal = self._resample_if_necessary(signal, sr)

        # Mix down the signal to mono
        signal = self._mix_down_if_necessary(signal)

        # Mel spectrogram transformation
        spectrogram = self.transformation(signal)
        return spectrogram[0], signal


    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.target_sample_rate)#.to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal



if __name__ == "__main__":
    all_files = _list_wav_files_recursively('../../dataset/youtube_clips')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=constants.SAMPLE_RATE,
        n_fft=constants.FILTER_LENGTH,
        hop_length=constants.HOP_LENGTH,
        n_mels=constants.NUM_CHANNELS,
        win_length=constants.WIN_LENGTH,
        f_min=constants.FMIN,
        f_max=constants.FMAX
    )

    # GET MEL
    stft = TacotronSTFT(constants.FILTER_LENGTH, constants.HOP_LENGTH, constants.WIN_LENGTH,
                        constants.NUM_CHANNELS, constants.SAMPLE_RATE,
                        constants.FMIN, constants.FMAX, center=False, device="cpu")
    wavpath = "../../dataset/youtube_clips/'NOBODY'.wav"
    melpath = wavpath.replace('.wav', '.mel')
    sr, wav = read_wav_np(wavpath)

    wav = torch.from_numpy(wav).unsqueeze(0)
    mel = stft.mel_spectrogram(wav)

    mel = mel.squeeze(0)


    dataset = AudioDataset(
        all_files,
        constants.SAMPLE_RATE,
        constants.TARGET_SAMPLES,
        mel_spectrogram,
        device="cuda"
    )

    print("MINE:",dataset[0][0].shape)
    print("THEIRS:",mel.shape)