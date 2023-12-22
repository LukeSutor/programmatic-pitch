import torch

## General
RUN_NAME = "training_01"
SEED = 2023
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA = 'dataset/data'
VALID_DATA = 'dataset/data'
NUM_WORKERS = 1
USE_AMP = True

## Logging
CHECKPOINT_DIR = 'chkpt'
LOG_DIR = 'logs'
SAMPLE_INTERVAL = 10
SAMPLE_NUMBER = 1
SUMMARY_INTERVAL = 1
VALIDATION_INTERVAL = 1
SAVE_INTERVAL = 1
DUPLICATE_TENSORBOARD = False

## Mel Transform
NUM_CHANNELS = 96
SAMPLE_RATE = 22050
TARGET_SAMPLES = 1024
HOP_LENGTH = 256 # WARNING: this can't be changed.
FILTER_LENGTH = 1024
SEGMENT_LENGTH = 16384 # Should be multiple of 256
PAD_SHORT = 2000
WIN_LENGTH = 1024
FMIN = 0
FMAX = 12000.0

## Diffusion Specific
# Training
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
DIFFUSION_LR = 1e-4
DIFFUSION_CHECKPOINT_PATH = 'C:/Users/Luke/Desktop/coding/diffusion_music_generation/chkpt/training_01/diffusion/0002.pt'
# Model
DIM = 32
DIM_MULTS = (1, 2, 4, 8)
CHANNELS = 1
TIMESTEPS = 1000
LOSS_TYPE = 'l1' # 'l1' or 'l2'
#EMA
EMA_DECAY = 0.995
STEP_START_EMA = 2000
UPDATE_EMA_EVERY = 10

## Univnet Specific
# Training
BATCH_SIZE = 1
OPTIMIZER = 'adam'
UNIVNET_CHECKPOINT_PATH = None
# Adam
UNIVNET_LR = 0.0001
BETA1 = 0.5
BETA2 = 0.9
STFT_LAMB = 2.5
#Generator
NOISE_DIM = 64
CHANNEL_SIZE = 16
DILATIONS = [1, 3, 9, 27]
STRIDES = [8, 8, 4]
GEN_LRELU_SLOPE = 0.2
KPNET_CONV_SIZE = 3
#mpd
PERIODS = [2,3,5,7,11]
KERNEL_SIZE = 5
STRIDE = 3
MPD_USE_SPECTRAL_NORM = False
MPD_LRELU_SLOPE = 0.2
#mrd
RESOLUTIONS = "[(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]" # (filter_length, hop_length, win_length)
MRD_USE_SPECTRAL_NORM = False
MRD_LRELU_SLOPE = 0.2
#Dist Config
DIST_BACKEND = "nccl"
DIST_URL = "tcp://localhost:54321"
WORLD_SIZE = 1