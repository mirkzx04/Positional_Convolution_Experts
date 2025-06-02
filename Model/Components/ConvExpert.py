from torch import nn as  nn

class ConvExpert(nn.Module):
    def __init__(self, kernel_size, out_channel):
        super().__init__()

        # Define experts operation
        self.pools = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=out_channel, kernel_size=kernel_size, padding=4),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channel, out_channels=out_channel * 2, kernel_size=kernel_size, padding=4),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.pools(X)