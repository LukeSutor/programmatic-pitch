import sys
import torch
from torchinfo import summary

sys.path.insert(0, 'C:/Users/Luke/Desktop/coding/diffusion_music_generation/models')
from denoising_diffusion import Unet, GaussianDiffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4


model = Unet(
    dim = 64,
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

diffusion.to(DEVICE)

summary(diffusion, input_size=(BATCH_SIZE, 2, 128, 1024))
# summary(diffusion, input_size=(BATCH_SIZE, 2, 128, 128))


# sampled_images = diffusion.sample(batch_size = 4)
# sampled_images.shape # (4, 3, 128, 128)

# if __name__ == "__main__":
