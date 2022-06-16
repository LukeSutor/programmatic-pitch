import os
import torchaudio
import torch
import random

SAMPLE_RATE = 44100
DEVICE = 'cuda'

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


def Transform(audio_dir, device=DEVICE):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    ).to(device)

    audio_data = load_info(audio_dir)
    duration = audio_data["samples"]
    sample_length = audio_data["samplerate"] * 30
    print(audio_data["samplerate"])
    start = random.randint(0, duration - sample_length)

    signal, sr = torchaudio.load(audio_dir, num_frames = sample_length, frame_offset = start)
    signal = signal.to(device)

    # Resample to make all sample rates the same
    signal = _resample_if_necessary(signal, sr, SAMPLE_RATE, device)

    # Make the signal mono if it isn't already
    # signal = _mix_down_if_necessary(signal)


    # Mel spectrogram transformation
    signal = mel_spectrogram(signal)
    return signal


def _resample_if_necessary(signal, sr, target_sample_rate, device):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            sr, target_sample_rate).to(device)
        signal = resampler(signal)
    return signal


def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

if __name__ == '__main__': # dataset\youtube_clips\Miramare x Clément Matrat – Foam.wav
    signal = Transform('dataset/youtube_clips/Miramare x Clément Matrat – Foam.wav')
    print(signal.shape)