import sys
import torch
from torchinfo import summary

sys.path.insert(0, '../')
import constants
from models.diffusion.denoising_diffusion import Unet, GaussianDiffusion, Trainer


if __name__ == "__main__":

    model = Unet(
        dim = constants.DIM, # 256
        dim_mults = constants.DIM_MULTS,
        channels = constants.CHANNELS
    )

    diffusion = GaussianDiffusion(
        denoise_fn = model,
        n_mels = constants.NUM_CHANNELS,
        n_samples = constants.TARGET_SAMPLES,
        channels = constants.CHANNELS,
        timesteps = constants.TIMESTEPS,   # number of steps
        loss_type = constants.LOSS_TYPE    # L1 or L2

    )

    summary(diffusion, input_size=(constants.BATCH_SIZE, 1, 96, 1024))


    # trainer = Trainer(
    #     diffusion,
    #     constants.TRAIN_DATA,
    #     constants.SAMPLE_RATE,
    #     constants.TARGET_SAMPLES,
    #     constants.DEVICE,
    #     train_batch_size = constants.BATCH_SIZE,
    #     train_num_steps = constants.TRAIN_STEPS,
    #     gradient_accumulate_every = constants.GRADIENT_ACCUMULATION,
    #     save_and_sample_every = constants.SAVE_SAMPLES_EVERY
    # )

    # trainer.train()
