import torch
import torch.nn as nn

from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator()
        self.MPD = MultiPeriodDiscriminator()

    def forward(self, x):
        return self.MRD(x), self.MPD(x)

if __name__ == '__main__':
    model = Discriminator()

    x = torch.randn(3, 1, 16384)
    print(x.shape)

    mrd_output, mpd_output = model(x)
    for features, score in mpd_output:
        for feat in features:
            print(feat.shape)
        print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
