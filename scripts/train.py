import sys
import torch
from torchinfo import summary

sys.path.insert(0, 'C:/Users/Luke/Desktop/coding/diffusion_music_generation/models')
from denoising_diffusion import Unet, GaussianDiffusion, Trainer

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FOLDER = '../dataset/youtube_clips'
BATCH_SIZE = 4
SAMPLE_RATE = 16000
TARGET_SAMPLES = 1024
GRADIENT_ACCUMULATION = 4
SAVE_SAMPLES_EVERY = 10000

if __name__ == "__main__":

    model = Unet(
        dim = 256,
        dim_mults = (1, 2, 4, 8),
        channels = 2
    )

    diffusion = GaussianDiffusion(
        denoise_fn = model,
        image_size = 128,
        n_mels = 128,
        n_samples = 1024,
        channels = 2,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2

    )

    summary(diffusion, input_size=(BATCH_SIZE, 2, 128, 1024))


    trainer = Trainer(
        diffusion,
        DATA_FOLDER,
        SAMPLE_RATE,
        TARGET_SAMPLES,
        DEVICE,
        train_batch_size = BATCH_SIZE,
        gradient_accumulate_every = GRADIENT_ACCUMULATION,
        save_and_sample_every = SAVE_SAMPLES_EVERY
    )

    trainer.train()
