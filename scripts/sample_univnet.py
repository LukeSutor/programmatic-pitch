import sys
import os
import torch
import torchaudio

# Add root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR+'/../'))
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
    model_g = Generator().to(constants.DEVICE)

    checkpoint = torch.load(model_path, map_location=constants.DEVICE)
    model_g.load_state_dict(checkpoint['model_g'])

    mel = mel.squeeze(0)
    mel = mel.to(constants.DEVICE)
    noise = torch.randn(constants.BATCH_SIZE, constants.NOISE_DIM, mel.size(2)).to(constants.DEVICE)

    with torch.no_grad():
        audio = model_g(mel, noise).squeeze(1)

    torchaudio.save(output_path, audio.cpu(), constants.SAMPLE_RATE)

if __name__ == "__main__":
    mel = get_mel(path='results/mel_47060_0.pt')

    sample(os.getcwd()+'/weights/0600.pt', 'results/sample_3.wav', mel)