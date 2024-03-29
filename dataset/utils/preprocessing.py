# Save all the data under 16Khz samplerate and separate into 80/20 train test split
import random
import os
import torch
import torchaudio

DATA_FOLDER = "D:/datasets/lofi_dataset/beats"
TRAIN_FOLDER = "D:/datasets/lofi_minimized/train"
VALID_FOLDER = "D:/datasets/lofi_minimized/valid"
TARGET_SAMPLE_RATE = 22050

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
        signal, sr = torchaudio.load(os.path.join(DATA_FOLDER, file))

        # Resample to make all sample rates the same
        signal = resample_if_necessary(signal, sr)

        # Mix down the signal to mono
        signal = mix_down_if_necessary(signal)

        # Save signal in train or valid folder
        filename = "{:04.0f}".format(i) + ".wav"
        if(random.random() > 0.2):
            # Train split
            torchaudio.save(os.path.join(TRAIN_FOLDER, filename), signal, TARGET_SAMPLE_RATE)
        else:
            # Valid split
            torchaudio.save(os.path.join(VALID_FOLDER, filename), signal, TARGET_SAMPLE_RATE)

        print("{:5.2f}% Complete".format((i + 1) / total_files * 100))

