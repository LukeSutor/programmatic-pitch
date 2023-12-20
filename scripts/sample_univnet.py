import sys
import os
import torch
import torchaudio
# Project-specific imports
sys.path[0] += '/../'
from models.univnet.generator import Generator
import constants

def get_mel(mel=None, path=None):
    '''
    Given either a path to a mel spectrogram or a mel spectrogram tensor, return the mel tensor

    Args:
        mel (torch.Tensor): mel spectrogram tensor
        path (str): path to mel spectrogram tensor
    '''
    if mel is None:
        mel = torch.load(path, map_location=constants.DEVICE)
    return mel

def sample(model_path, output_path, mel):
    '''
    Generate waveform sample from a trained model

    Args:
        model_path (str): path to trained model
        input_path (str): path to input mel spectrogram
        output_path (str): path to save generated waveform
    '''
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_g = Generator().to(constants.DEVICE)

    checkpoint = torch.load(model_path, map_location=constants.DEVICE)
    model_g.load_state_dict(checkpoint['model_g'])

    mel = mel.unsqueeze(0)
    mel = mel .to(constants.DEVICE)
    noise = torch.randn(constants.BATCH_SIZE, constants.NOISE_DIM, mel.size(2)).to(constants.DEVICE)

    with torch.no_grad():
        audio = model_g(mel, noise).squeeze(1)

    torchaudio.save(output_path, audio.cpu(), constants.SAMPLE_RATE)

    pass

if __name__ == "__main__":
    mel = get_mel(path=os.getcwd()+'/../test.pt')

    sample(os.getcwd()+'/../weights/univnet_final.pt', 'sample.wav', mel)