import os
import sys
import time
import logging
import argparse
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

sys.path.insert(0, '../')
import constants
from utils.univnet.train import train

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    assert constants.HOP_LENGTH == 256, \
        'constants.HOP_LENGTH must be equal to 256, got %d' % constants.HOP_LENGTH

    num_gpus = 0
    torch.manual_seed(constants.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(constants.SEED)
        num_gpus = torch.cuda.device_count()
        print('Batch size per GPU :', constants.BATCH_SIZE)
    else:
        pass

    if num_gpus > 1:
        mp.spawn(train, num_gpus, nprocs=num_gpus)
    else:
        train(0, num_gpus)