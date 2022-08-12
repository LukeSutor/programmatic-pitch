import math
import random

import blobfile as bf
# from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
import soundfile as sf


def create_dataloader(hp, args, train, device):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=hp.audio.filter_length,
        hop_length=hp.audio.hop_length,
        n_mels=hp.audio.n_mel_channels,
        win_length=hp.audio.win_length,
        f_min=hp.audio.mel_fmin,
        f_max=hp.audio.mel_fmax
    )

    if train:
        dataset = AudioDataset(hp.data.train_dir, hp.audio.sampling_rate, hp.audio.target_samples, mel_spectrogram, device="cuda")
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
                          num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)

    else:
        dataset = AudioDataset(hp.data.val_dir, hp.audio.sampling_rate, hp.audio.target_samples, mel_spectrogram, device="cuda")
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)


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
        sample_length = math.floor(512 * 1023 * audio_data["samplerate"] / self.target_sample_rate)
        start = random.randint(0, duration - sample_length)

        signal, sr = torchaudio.load(path, num_frames = sample_length, frame_offset = start)
        # signal = signal.to(self.device)

        # Resample to make all sample rates the same
        signal = self._resample_if_necessary(signal, sr)

        # Mel spectrogram transformation
        spectrogram = self.transformation(signal)
        return spectrogram, signal


    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.target_sample_rate)#.to(self.device)
            signal = resampler(signal)
        return signal



if __name__ == "__main__":

    SAMPLE_RATE = 16000
    TARGET_SAMPLES = 1024

    all_files = _list_wav_files_recursively('../../dataset/youtube_clips')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )


    dataset = AudioDataset(
        all_files,
        SAMPLE_RATE,
        TARGET_SAMPLES,
        mel_spectrogram,
        device="cuda"
    )


    print(dataset[0])