import os
import copy
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
import itertools
from .writer import MyWriter

from models.diffusion.denoising_diffusion import Unet, GaussianDiffusion, EMA, Adam, GradScaler
from utils.dataloader import create_dataloader
import constants



def cycle(dl):
    while True:
        for data in dl:
            yield data

def reset_parameters(ema_model, model):
    ema_model.load_state_dict(model.state_dict())

def step_ema(step, ema, ema_model, model):
    if step < constants.STEP_START_EMA:
        reset_parameters(ema_model, model)
        return
    ema.update_model_average(ema_model, model)


def train(rank, num_gpus):

    with open('constants.py', 'r') as f:
        hyperparams = ''.join(f.readlines())

    device = torch.device('cuda:{:d}'.format(rank))

    # initialize models
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
    ).to(device)

    ema = EMA(constants.EMA_DECAY)

    ema_model = copy.deepcopy(diffusion)

    trainloader = create_dataloader(True, device='cpu')

    opt = Adam(diffusion.parameters(), lr = constants.DIFFUSION_LR)
               
    reset_parameters(ema_model, diffusion)
    scaler = GradScaler(enabled = constants.USE_AMP)

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    # Create directories and writer
    log_dir = os.path.join(constants.LOG_DIR, constants.RUN_NAME, 'diffusion')
    chkpt_dir = os.path.join(constants.CHECKPOINT_DIR, constants.RUN_NAME, 'diffusion')

    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(chkpt_dir, exist_ok=True)
        writer = MyWriter(log_dir)

    if constants.DIFFUSION_CHECKPOINT_PATH is not None:
        if rank == 0:
            print("Resuming from checkpoint: %s" % constants.DIFFUSION_CHECKPOINT_PATH)
        checkpoint = torch.load(constants.DIFFUSION_CHECKPOINT_PATH)
        init_epoch = checkpoint['epoch']
        diffusion.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema'])
        scaler.load_state_dict(checkpoint['scaler'])

        if rank == 0 and hyperparams != checkpoint['hyperparams']:
            print("New hyperparams are different from checkpoint. Will use new.")
    else:
        init_epoch = -1
        if rank == 0:
            print("Starting new training run.")

    if num_gpus > 1:
        diffusion = DistributedDataParallel(diffusion, device_ids=[rank]).to(device)

    diffusion.train()

    for epoch in itertools.count(init_epoch+1):

        if rank == 0:
            loader = tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader

        for mel, _ in loader:
            mel = mel.unsqueeze(0).to(device)

            with autocast(enabled = constants.USE_AMP):
                    loss = diffusion(mel)
                    scaler.scale(loss / constants.GRADIENT_ACCUMULATION).backward()

            if rank == 0:
                loader.set_description(f'loss: {loss.item():.4f}')

                # Log training
                if epoch % constants.SUMMARY_INTERVAL == 0:
                    writer.log_training(loss.item(), epoch)

            if epoch != 0 and epoch % constants.GRADIENT_ACCUMULATION == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            if epoch % constants.UPDATE_EMA_EVERY == 0:
                step_ema(epoch, ema, ema_model, diffusion)

            # Sample model
            if epoch != 0 and epoch % constants.SAMPLE_INTERVAL == 0:
                ema_model.eval()

                milestone = epoch // constants.SAMPLE_INTERVAL

                for i in range(constants.SAMPLE_NUMBER):
                        image = ema_model.sample()
                        torch.save(image, f'mel_{milestone}_{i}.pt')
                        if rank == 0 and i == 0:
                            writer.log_mel_spec(image.squeeze(0).squeeze(0).cpu().detach().numpy(), epoch)      
            # Save checkpoint
            if epoch != 0 and epoch % constants.SAVE_INTERVAL == 0:
                save_path = os.path.join(chkpt_dir, f'%04d.pt' % epoch)
                torch.save({
                    'epoch': epoch,
                    'model': diffusion.state_dict(),
                    'ema': ema_model.state_dict(),
                    'scaler': scaler.state_dict(),
                    'hyperparams': hyperparams,
                }, save_path)