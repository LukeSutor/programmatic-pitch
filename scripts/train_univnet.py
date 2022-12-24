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
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    assert constants.HOP_LENGTH == 256, \
        'constants.HOP_LENGTH must be equal to 256, got %d' % constants.HOP_LENGTH

    args.num_gpus = 0
    torch.manual_seed(constants.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(constants.SEED)
        args.num_gpus = torch.cuda.device_count()
        print('Batch size per GPU :', constants.BATCH_SIZE)
    else:
        pass

    if args.num_gpus > 1:
        mp.spawn(train, nprocs=args.num_gpus,
                 args=(args, args.checkpoint_path))
    else:
        train(0, args, args.checkpoint_path)