import torch

## General
RUN_NAME = "Testing"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA = '../dataset/data/train'
VALID_DATA = '../dataset/data/valid'
NUM_WORKERS = 1

## Mel Transform
NUM_CHANNELS = 128
SAMPLE_RATE = 16000
TARGET_SAMPLES = 1024
HOP_LENGTH = 256 # WARNING: this can't be changed.
FILTER_LENGTH = 1024
SEGMENT_LENGTH = 16384 # Should be multiple of 256
PAD_SHORT = 2000
WIN_LENGTH = 1024
FMIN = 0
FMAX = 12000.0

## Diffusion Specific
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
TRAIN_STEPS = 100000
SAVE_SAMPLES_EVERY = 10000

## Univnet Specific
# Training
BATCH_SIZE = 1
OPTIMIZER = 'adam'
SEED = 2022
CHECKPOINT_PATH = None
# Adam
LR = 0.0001
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
#Log
SUMMARY_INTERVAL = 1
VALIDATION_INTERVAL = 100
SAVE_INTERVAL = 100
NUM_AUDIO = 5
CHECKPOINT_DIR = 'chkpt'
LOG_DIR = 'logs'

