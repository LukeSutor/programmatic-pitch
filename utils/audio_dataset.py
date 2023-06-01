import math
import random
import os

import blobfile as bf
# from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
import soundfile as sf

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

def load_data(
    *,
    data_dir,
    batch_size,
    audio_length,
    target_sample_rate,
    transformation,
    class_cond=False,
    deterministic=False,
    device
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.
    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param audio_length: the length in seconds of audio signals to be transformed.
    :param target_sample_rate: the sample rate to transform audio signals into.
    :param transformation: the transformation (mel spectrogram) to perform on the data.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param device: device to perform transformations and save data to.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = os.listdir(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = AudioDataset(
        audio_length,
        all_files,
        target_sample_rate,
        transformation,
        device=device
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


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

        # Mel spectrogram transformation
        signal = self.transformation(signal)
        return signal


    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.target_sample_rate)#.to(self.device)
            signal = resampler(signal)
        return signal


if __name__ == "__main__":

    SAMPLE_RATE = 16000
    TARGET_SAMPLES = 1024

    all_files = os.path.join('D:/datasets/lofi')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        win_length=1024,
        f_min=0.0,
        f_max=12000.0
    )


    dataset = AudioDataset(
        all_files,
        SAMPLE_RATE,
        TARGET_SAMPLES,
        mel_spectrogram,
        device="cuda"
    )


    print(len(dataset))
    # torch.Size([2, 128, 1024])

    # data = np.array(dataset[2])


    # sf.write("16kHz.wav", data[0], SAMPLE_RATE)