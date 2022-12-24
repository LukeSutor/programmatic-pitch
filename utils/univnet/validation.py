import sys
import tqdm
import torch
import torchaudio
import torch.nn.functional as F
import constants

def validate(generator, discriminator, valloader, stft, writer, step, device):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=constants.SAMPLE_RATE,
        n_fft=constants.FILTER_LENGTH,
        hop_length=constants.HOP_LENGTH,
        n_mels=constants.NUM_CHANNELS,
        win_length=constants.WIN_LENGTH,
        f_min=constants.FMIN,
        f_max=constants.FMAX
    )

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    mel_loss = 0.0
    for idx, (mel, audio) in enumerate(loader):
        mel = mel.to(device)
        audio = audio.to(device)
        noise = torch.randn(1, constants.NOISE_DIM, mel.size(2)).to(device)

        fake_audio = generator(mel, noise)[:,:,:audio.size(2)]

        # mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
        # mel_real = stft.mel_spectrogram(audio.squeeze(1))

        mel_fake = mel_spectrogram(fake_audio.squeeze(1))
        mel_real = mel_spectrogram(audio.squeeze(1))

        print("SHAPE:", mel_fake.shape)

        mel_loss += F.l1_loss(mel_fake, mel_real).item()

        if idx < constants.NUM_AUDIO:
            spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
            spec_real = stft.linear_spectrogram(audio.squeeze(1))

            audio = audio[0][0].cpu().detach().numpy()
            fake_audio = fake_audio[0][0].cpu().detach().numpy()
            spec_fake = spec_fake[0].cpu().detach().numpy()
            spec_real = spec_real[0].cpu().detach().numpy()
            writer.log_fig_audio(audio, fake_audio, spec_fake, spec_real, idx, step)

    mel_loss = mel_loss / len(valloader.dataset)

    writer.log_validation(mel_loss, generator, discriminator, step)

    torch.backends.cudnn.benchmark = True