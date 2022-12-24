import sys
import torch
from torchinfo import summary

sys.path.insert(0, '../')
import constants
from models.diffusion.denoising_diffusion import Unet, GaussianDiffusion, Trainer


if __name__ == "__main__":

    model = Unet(
        dim = 16, # 256
        dim_mults = (1, 2, 4, 8),
        channels = 2
    )

    diffusion = GaussianDiffusion(
        denoise_fn = model,
        image_size = 128,
        n_mels = 128,
        n_samples = 1024, # 1024
        channels = 2,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2

    )

    # summary(diffusion, input_size=(constants.BATCH_SIZE, 2, 128, 512))


    trainer = Trainer(
        diffusion,
        constants.TRAIN_DATA,
        constants.SAMPLE_RATE,
        constants.TARGET_SAMPLES,
        constants.DEVICE,
        train_batch_size = constants.BATCH_SIZE,
        train_num_steps = constants.TRAIN_STEPS,
        gradient_accumulate_every = constants.GRADIENT_ACCUMULATION,
        save_and_sample_every = constants.SAVE_SAMPLES_EVERY
    )

    trainer.train()
