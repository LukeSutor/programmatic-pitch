# Save all the data under 16Khz samplerate and separate into 80/20 train test split
import random
import os
import torch
import torchaudio

DATA_FOLDER = '../youtube_clips/'
TRAIN_FOLDER = '../data/train/'
VALID_FOLDER = '../data/valid/'
TARGET_SAMPLE_RATE = 16000

def resample_if_necessary(signal, sr):
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sr, TARGET_SAMPLE_RATE)#.to(self.device)
        signal = resampler(signal)
    return signal

def mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


if __name__ == "__main__":
    total_files = len(os.listdir(DATA_FOLDER))

    for i, file in enumerate(os.listdir(DATA_FOLDER)):
        signal, sr = torchaudio.load(DATA_FOLDER + file)

        signal

        # Resample to make all sample rates the same
        signal = resample_if_necessary(signal, sr)

        # Mix down the signal to mono
        signal = mix_down_if_necessary(signal)

        # Save signal in train or valid folder
        if(random.random() > 0.2):
            # Train split
            torchaudio.save(TRAIN_FOLDER + file, signal, TARGET_SAMPLE_RATE)
        else:
            # Valid split
            torchaudio.save(VALID_FOLDER + file, signal, TARGET_SAMPLE_RATE)

        print("{:3.2f}% Complete".format(i / total_files))

