import torch.nn as nn


class DownsampleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act   = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

        self.skip  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        )

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(y + self.skip(x))
