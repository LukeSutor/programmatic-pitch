import sys
import math
import random
from datetime import datetime

import torch
import os
import librosa
from pedalboard import Pedalboard, Reverb, PitchShift, Invert
from torch.utils.data import DataLoader, Dataset
import torchaudio

# Project-specific imports
sys.path.insert(1, ".")
from utils.univnet.stft import TacotronSTFT
import constants


def create_dataloader(train, device):
    if train:
        dataset = AudioDataset(constants.TRAIN_DATA, constants.SAMPLE_RATE, constants.TARGET_SAMPLES, True, device="cuda")
        return DataLoader(dataset=dataset, batch_size=constants.BATCH_SIZE, shuffle=True,
                          num_workers=constants.NUM_WORKERS, pin_memory=True, drop_last=True)
    else:
        dataset = AudioDataset(constants.VALID_DATA, constants.SAMPLE_RATE, constants.TARGET_SAMPLES, False, device="cuda")
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=constants.NUM_WORKERS, pin_memory=True, drop_last=False)


def load_info(path: str, filename: str) -> dict:
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
    si = torchaudio.info(os.path.join(path, filename))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info


class AudioDataset(Dataset):
    def __init__(
        self,
        audio_path,
        target_sample_rate,
        target_samples,
        train,
        device="cpu"
    ):
        super().__init__()
        self.audio_path = audio_path
        self.local_images = os.listdir(audio_path)
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.target_samples = target_samples
        self.train = train
        self.transformation = TacotronSTFT(filter_length=constants.FILTER_LENGTH,
                                            hop_length=constants.HOP_LENGTH,
                                            win_length=constants.WIN_LENGTH,
                                            n_mel_channels=constants.NUM_CHANNELS,
                                            sampling_rate=constants.SAMPLE_RATE,
                                            mel_fmin=constants.FMIN,
                                            mel_fmax=constants.FMAX,
                                            center=False,
                                            device="cpu")
        
        # Seed random number generator
        random.seed(datetime.now().timestamp())
        

    def __len__(self):
        return len(self.local_images)


    def __getitem__(self, idx):
        filename = self.local_images[idx]

        # Get a random sequence from the data of length sr * audio_length
        audio_data = load_info(self.audio_path, filename)
        duration = audio_data["samples"]
        sample_length = math.floor(constants.HOP_LENGTH * constants.TARGET_SAMPLES * audio_data["samplerate"] / self.target_sample_rate)
        # Add extra 2 second padding to the sample length so that augmentations can speed up the audio
        sample_length += 3 * audio_data["samplerate"]
        start = random.randint(0, duration - sample_length)

        signal, sr = torchaudio.load(os.path.join(self.audio_path, filename), num_frames = sample_length, frame_offset = start)
        signal.to("cpu")

        # Make the signal mono
        signal = self._mix_down_if_necessary(signal)

        # Augment the signal if in training
        if self.train:
            signal = self._augment(signal, sr)

        # Resample the signal
        signal = self._resample_if_necessary(signal, sr)

        # Reshape signal to proper length
        signal = signal.narrow(1, 0, constants.HOP_LENGTH * constants.TARGET_SAMPLES)

        # # Resampling and mixing down may cause floating point errors, so reclamp values
        # signal = signal.clamp(-1.0, 1.0)

        # Mel spectrogram transformation
        spectrogram = self.transformation.mel_spectrogram(signal).squeeze(0)
        return spectrogram, signal


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
    

    def _augment(self, signal, sr):
        '''
        Performs augmentations on the signal if it's for the training set
        Every augmentation has a 30% chance of being added
        Possible augmentations:
            - Speed Change
            - Pitch Shift
            - Reverb
            - Inverse
        '''
        effects = [(1 if random.random() < 0.3 else 0) for _ in range(4)]

        if effects.count(1) == 0:
            # Crop signal to correct size
            return signal

        np_signal = signal.detach().cpu().numpy()

        # Add speed change in range [0.8, 1.2]
        if effects[0]:
            factor = random.random() * 0.3 + 0.85
            np_signal = librosa.effects.time_stretch(np_signal, rate=factor)

        if effects[1:].count(1) > 0:
            effects_list = []
            # Add effects to spotify pedalboard
            if effects[1]:
                # Semitones are in range [-4, 4]
                effects_list.append(PitchShift(random.randint(-4, 4)))
            if effects[2]:
                rs = random.random() * 0.35 + 0.05
                wl = random.random() * 0.2 + 0.05
                dl = random.random() * 0.3 + 0.05
                effects_list.append(Reverb(room_size=rs, wet_level=wl, dry_level=dl))
            if effects[3]:
                effects_list.append(Invert())
            
            board = Pedalboard(effects_list)
            np_signal = board(np_signal, sr)

        signal = torch.from_numpy(np_signal)

        return torch.from_numpy(np_signal)
        


if __name__ == "__main__":
    # from pedalboard.io import AudioFile
    # with AudioFile(os.path.join(constants.TRAIN_DATA, "Sarcastic Sounds - Hurt Me.wav")).resampled_to(16000) as f:
    #     audio = f.read(f.frames)
    # print(type(audio))


    dataset = AudioDataset(constants.TRAIN_DATA, constants.SAMPLE_RATE, constants.TARGET_SAMPLES, True, device="cuda")
    
    print("Shape:", dataset[0][0].shape) # torch.Size([100, 1024])

    

    # for i in range(len(dataset)):
    #     spectrogram, signal = dataset[0]
    #     torchaudio.save(f'test_{i}.wav', signal, constants.SAMPLE_RATE)