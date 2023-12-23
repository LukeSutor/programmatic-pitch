from torch.utils.tensorboard import SummaryWriter
import numpy as np
import librosa

from ..plotting import plot_spectrogram_to_numpy


class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, loss, step):
        self.add_scalar('loss', loss, step)
        
    def log_mel_spec(self, mel, step):
        mel = librosa.amplitude_to_db(mel, ref=np.max,top_db=80.)
        self.add_image('mel_spectrogram', plot_spectrogram_to_numpy(mel), step)