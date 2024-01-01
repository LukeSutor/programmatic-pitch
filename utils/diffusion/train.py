import os
import copy
import shutil
import torch
from torch.optim import Adam
from accelerate import Accelerator
from tqdm import tqdm
import itertools
from .writer import MyWriter

from models.diffusion.denoising_diffusion import Unet, GaussianDiffusion
from models.diffusion.EMA import EMA
from utils.dataloader import create_dataloader
from utils.diffusion.validation import validate
import constants


def reset_parameters(ema_model, model):
    ema_model.load_state_dict(model.state_dict())

def step_ema(step, ema, ema_model, model):
    if step < constants.STEP_START_EMA:
        reset_parameters(ema_model, model)
        return
    ema.update_model_average(ema_model, model)


def train():

    with open('constants.py', 'r') as f:
        hyperparams = ''.join(f.readlines())

    accelerator = Accelerator(
        mixed_precision = 'fp16' if constants.USE_AMP else 'no',
    )
    device = accelerator.device

    # initialize models
    model = Unet(
        dim = constants.DIM,
        dim_mults = constants.DIM_MULTS,
        channels = constants.CHANNELS
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = (constants.NUM_CHANNELS, constants.TARGET_SAMPLES),
        timesteps = constants.TIMESTEPS,   # number of steps
        sampling_timesteps=4
    ).to(device)

    ema = EMA(constants.EMA_DECAY)

    ema_model = copy.deepcopy(diffusion)

    trainloader = create_dataloader(True, device='cpu')
    valloader = create_dataloader(False, device='cpu')

    opt = Adam(diffusion.parameters(), lr = constants.DIFFUSION_LR)
               
    diffusion, opt, trainloader = accelerator.prepare(diffusion, opt, trainloader)

    reset_parameters(ema_model, diffusion)

    # Create directories and writer
    log_dir = os.path.join(constants.LOG_DIR, constants.RUN_NAME, 'diffusion')
    chkpt_dir = os.path.join(constants.CHECKPOINT_DIR, constants.RUN_NAME, 'diffusion')
    results_dir = os.path.join(constants.RESULTS_DIR, constants.RUN_NAME, 'diffusion')

    if accelerator.is_local_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(chkpt_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        writer = MyWriter(log_dir)

    if constants.DIFFUSION_CHECKPOINT_PATH is not None:
        accelerator.print("Resuming from checkpoint: %s" % constants.DIFFUSION_CHECKPOINT_PATH)
        checkpoint = torch.load(constants.DIFFUSION_CHECKPOINT_PATH)
        init_epoch = checkpoint['epoch']
        step = checkpoint['step']
        diffusion.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema'])
        if accelerator.scaler is not None and checkpoint['scaler'] is not None:
            accelerator.scaler.load_state_dict(checkpoint['scaler'])

        if hyperparams != checkpoint['hyperparams']:
            accelerator.print("New hyperparams are different from checkpoint. Will use new.")
    else:
        init_epoch = -1
        step = 0
        accelerator.print("Starting new training run.")

    diffusion.train()

    for epoch in itertools.count(init_epoch+1):

        total_loss = 0.0

        if accelerator.is_local_main_process:
            loader = tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader

        for mel, _ in loader:
            mel = mel.unsqueeze(1).to(device)

            with accelerator.autocast():
                loss = diffusion(mel)
                loss = loss / constants.GRADIENT_ACCUMULATION
                total_loss += loss.item()

            accelerator.backward(loss)

            loader.set_description(f'loss: {loss.item():.4f}')

            # Log training
            if step % constants.SUMMARY_INTERVAL == 0:
                writer.log_training(loss.item(), step)

            accelerator.wait_for_everyone()
            # accelerator.clip_grad_norm_(diffusion.parameters(), 1.0)

            if step != 0 and step % constants.GRADIENT_ACCUMULATION == 0:
                opt.step()
                opt.zero_grad()

            if step % constants.UPDATE_EMA_EVERY == 0:
                step_ema(step, ema, ema_model, diffusion)
            
            step += 1

        if accelerator.is_local_main_process:
            # Log epoch loss
            total_loss /= len(trainloader.dataset)
            writer.log_epoch(total_loss, epoch)

            # Validate model
            if epoch != 0 and epoch % constants.VALIDATION_INTERVAL == 0:
                with torch.no_grad():
                    validate(ema_model, valloader, writer, step, device)

            # Sample model
            if epoch != 0 and epoch % constants.SAMPLE_INTERVAL == 0:
                ema_model.eval()

                milestone = epoch // constants.SAMPLE_INTERVAL

                for i in range(constants.SAMPLE_NUMBER):
                        image = ema_model.sample(batch_size=1)
                        if i == 0:
                            os.makedirs(os.path.join(results_dir, str(milestone)), exist_ok=True)
                            writer.log_mel_spec(image.squeeze(0).squeeze(0).cpu().detach().numpy(), step)      
                        torch.save(image, os.path.join(results_dir, str(milestone), f'mel_{i}.pt'))

            # Save checkpoint
            if epoch != 0 and epoch % constants.SAVE_INTERVAL == 0:
                save_path = os.path.join(chkpt_dir, f'%04d.pt' % epoch)
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model': diffusion.state_dict(),
                    'ema': ema_model.state_dict(),
                    'scaler': accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
                    'hyperparams': hyperparams,
                }, save_path)

                # Create a copy of the tensorboard file to be downloaded
                if constants.DUPLICATE_TENSORBOARD:
                    for file in os.listdir(log_dir):
                        if file.endswith(".0") and file.count("copy") == 0:
                            shutil.copyfile(os.path.join(log_dir, file), os.path.join(log_dir, "events.out.tfevents.copy.0"))