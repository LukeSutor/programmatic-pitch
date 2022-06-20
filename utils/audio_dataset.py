import math
import random

import blobfile as bf
# from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
import torchaudio

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
    all_files = _list_wav_files_recursively(data_dir)
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
        classes=classes,
        # shard=MPI.COMM_WORLD.Get_rank(),
        # num_shards=MPI.COMM_WORLD.Get_size(),
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
        audio_length,
        image_paths,
        target_sample_rate,
        transformation,
        classes=None,
        shard=0,
        num_shards=1,
        device="cpu"
    ):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.audio_length = audio_length
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        

    def __len__(self):
        return len(self.local_images)


    def __getitem__(self, idx):
        path = self.local_images[idx]

        # Get a random sequence from the data of length sr * audio_length
        audio_data = load_info(path)
        duration = audio_data["samples"]
        sample_length = audio_data["samplerate"] * self.audio_length
        start = random.randint(0, duration - sample_length)

        signal, sr = torchaudio.load(path, num_frames = sample_length, frame_offset = start)
        signal = signal.to(self.device)

        # Resample to make all sample rates the same
        signal = self._resample_if_necessary(signal, sr)

        # Mel spectrogram transformation
        signal = self.transformation(signal)
        return signal


    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
