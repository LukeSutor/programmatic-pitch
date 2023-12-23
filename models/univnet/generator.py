import torch
import torch.nn as nn
from .lvcnet import LVCBlock

import constants


class Generator(nn.Module):
    """UnivNet Generator"""
    def __init__(self):
        super(Generator, self).__init__()
        self.mel_channel = constants.NUM_CHANNELS
        self.noise_dim = constants.NOISE_DIM
        self.hop_length = constants.HOP_LENGTH
        channel_size = constants.CHANNEL_SIZE
        kpnet_conv_size = constants.KPNET_CONV_SIZE

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in constants.STRIDES:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    constants.NUM_CHANNELS,
                    stride=stride,
                    dilations=constants.DILATIONS,
                    lReLU_slope=constants.GEN_LRELU_SLOPE,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )
        
        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(constants.NOISE_DIM, channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(constants.GEN_LRELU_SLOPE),
            nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

    def forward(self, c, z):
        '''
        Args: 
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length) 
            z (Tensor): the noise sequence (batch, noise_dim, in_length)
        
        '''
        z = self.conv_pre(z)                # (B, c_g, L)

        for res_block in self.res_stack:
            res_block.to(z.device)
            z = res_block(z, c)             # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)               # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, c, z=None):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(c.device)
        mel = torch.cat((c, zero), dim=2)
        
        if z is None:
            z = torch.randn(1, self.noise_dim, mel.size(2)).to(mel.device)

        audio = self.forward(mel, z)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(self.hop_length*10)]
        audio = constants.MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-constants.MAX_WAV_VALUE, max=constants.MAX_WAV_VALUE-1)
        audio = audio.short()

        return audio

if __name__ == '__main__':
    model = Generator()

    c = torch.randn(3, 100, 10)
    z = torch.randn(3, 64, 10)
    print(c.shape)

    y = model(c, z)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)