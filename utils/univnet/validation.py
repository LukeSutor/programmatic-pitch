import os
import sys
import tqdm
import torch
import torchaudio
import torch.nn.functional as F
import shutil

# Project-specific imports
sys.path.insert(1, ".")
import constants

def validate(generator, discriminator, valloader, stft, writer, step, device, log_dir):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    mel_loss = 0.0
    for idx, (mel, audio) in enumerate(loader):
        mel = mel.to(device)
        audio = audio.to(device)
        noise = torch.randn(1, constants.NOISE_DIM, mel.size(2)).to(device)

        fake_audio = generator(mel, noise)[:,:,:audio.size(2)]

        mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
        mel_real = stft.mel_spectrogram(audio.squeeze(1))

        mel_loss += F.l1_loss(mel_fake, mel_real).item()

        if idx < constants.SAMPLE_NUMBER:
            spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
            spec_real = stft.linear_spectrogram(audio.squeeze(1))

            audio = audio[0][0].cpu().detach().numpy()
            fake_audio = fake_audio[0][0].cpu().detach().numpy()
            spec_fake = spec_fake[0].cpu().detach().numpy()
            spec_real = spec_real[0].cpu().detach().numpy()
            writer.log_fig_audio(audio, fake_audio, spec_fake, spec_real, idx, step)

    mel_loss = mel_loss / len(valloader.dataset)

    writer.log_validation(mel_loss, generator, discriminator, step)

    # Create a copy of the tensorboard file to be downloaded
    for file in os.listdir(log_dir):
        if file.endswith(".0") and file.count("copy") == 0:
            shutil.copyfile(os.path.join(log_dir, file), os.path.join(log_dir, "events.out.tfevents.copy.0"))

    torch.backends.cudnn.benchmark = True