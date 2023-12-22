import os
import sys
import time
import shutil
import logging
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
import itertools

from .writer import MyWriter
from .stft import TacotronSTFT
from .stft_loss import MultiResolutionSTFTLoss
from ..dataloader import create_dataloader

# Project-specific imports
from models.univnet.generator import Generator
from models.univnet.discriminator import Discriminator
from .validation import validate
import constants


def train(rank, num_gpus):

    with open('constants.py', 'r') as f:
        hyperparams = ''.join(f.readlines())

    if num_gpus > 1:
        init_process_group(backend=constants.DIST_BACKEND, init_method=constants.DIST_URL,
                           world_size=constants.WORLD_SIZE * num_gpus, rank=rank)

    torch.cuda.manual_seed(constants.SEED)
    device = torch.device('cuda:{:d}'.format(rank))

    model_g = Generator().to(device)
    model_d = Discriminator().to(device)

    optim_g = torch.optim.AdamW(model_g.parameters(),
        lr=constants.UNIVNET_LR, betas=(constants.BETA1, constants.BETA2))
    optim_d = torch.optim.AdamW(model_d.parameters(),
        lr=constants.UNIVNET_LR, betas=(constants.BETA1, constants.BETA2))

    init_epoch = -1
    step = 0
    log_dir = os.path.join(constants.LOG_DIR, constants.RUN_NAME, 'univnet')
    chkpt_dir = os.path.join(constants.CHECKPOINT_DIR, constants.RUN_NAME, 'univnet')

    # define logger, writer, valloader, stft at rank_zero
    if rank == 0:
        pt_dir = os.path.join(constants.CHECKPOINT_DIR, constants.RUN_NAME)
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (constants.RUN_NAME, time.time()))),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        writer = MyWriter(log_dir)
        valloader = create_dataloader(False, device='cpu')
        stft = TacotronSTFT(filter_length=constants.FILTER_LENGTH,
                            hop_length=constants.HOP_LENGTH,
                            win_length=constants.WIN_LENGTH,
                            n_mel_channels=constants.NUM_CHANNELS,
                            sampling_rate=constants.SAMPLE_RATE,
                            mel_fmin=constants.FMIN,
                            mel_fmax=constants.FMAX,
                            center=False,
                            device=device)

    if constants.UNIVNET_CHECKPOINT_PATH is not None:
        if rank == 0:
            logger.info("Resuming from checkpoint: %s" % constants.UNIVNET_CHECKPOINT_PATH)
        checkpoint = torch.load(constants.UNIVNET_CHECKPOINT_PATH)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if rank == 0:
            if hyperparams != checkpoint['hyperparams']:
                logger.warning("New hyperparams are different from checkpoint. Will use new.")

    else:
        if rank == 0:
            logger.info("Starting new training run.")

    if num_gpus > 1:
        model_g = DistributedDataParallel(model_g, device_ids=[rank]).to(device)
        model_d = DistributedDataParallel(model_d, device_ids=[rank]).to(device)

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    trainloader = create_dataloader(True, device='cpu')

    model_g.train()
    model_d.train()

    resolutions = eval(constants.RESOLUTIONS)
    stft_criterion = MultiResolutionSTFTLoss(device, resolutions)

    for epoch in itertools.count(init_epoch+1):
        
        if rank == 0 and epoch % constants.VALIDATION_INTERVAL == 0:
            with torch.no_grad():
                validate(model_g, model_d, valloader, stft, writer, step, device, log_dir)

        if rank == 0:
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader

        for mel, audio in loader:

            mel = mel.to(device)
            audio = audio.to(device)
            noise = torch.randn(constants.BATCH_SIZE, constants.NOISE_DIM, mel.size(2)).to(device)

            # generator
            optim_g.zero_grad()
            fake_audio = model_g(mel, noise)

            # Multi-Resolution STFT Loss
            sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * constants.STFT_LAMB

            res_fake, period_fake = model_d(fake_audio)

            score_loss = 0.0

            for (_, score_fake) in res_fake + period_fake:
                score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))

            score_loss = score_loss / len(res_fake + period_fake)

            loss_g = score_loss + stft_loss

            loss_g.backward()
            optim_g.step()

            # discriminator

            optim_d.zero_grad()
            res_fake, period_fake = model_d(fake_audio.detach())
            res_real, period_real = model_d(audio)

            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(res_fake + period_fake, res_real + period_real):
                loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
                loss_d += torch.mean(torch.pow(score_fake, 2))

            loss_d = loss_d / len(res_fake + period_fake)

            loss_d.backward()
            optim_d.step()

            step += 1
            # logging
            loss_g = loss_g.item()
            loss_d = loss_d.item()

            if rank == 0 and step % constants.SUMMARY_INTERVAL == 0:
                writer.log_training(loss_g, loss_d, stft_loss.item(), score_loss.item(), step)
                loader.set_description("g %.04f d %.04f | step %d" % (loss_g, loss_d, step))

        if rank == 0 and epoch % constants.SAVE_INTERVAL == 0:
            save_path = os.path.join(chkpt_dir, '%04d.pt' % epoch)
            torch.save({
                'model_g': (model_g.module if num_gpus > 1 else model_g).state_dict(),
                'model_d': (model_d.module if num_gpus > 1 else model_d).state_dict(),
                'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
                'step': step,
                'epoch': epoch,
                'hyperparams': hyperparams,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)

            # Create a copy of the tensorboard file to be downloaded
            for file in os.listdir(log_dir):
                if file.endswith(".0") and file.count("copy") == 0:
                    shutil.copyfile(os.path.join(log_dir, file), os.path.join(log_dir, "events.out.tfevents.copy.0"))