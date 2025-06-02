from torch import nn as  nn

class ConvExpert(nn.Module):
    def __init__(self, kernel_size, out_channel):
        super().__init__()

        # Define experts operation
        self.pools = nn.ModuleList(
            nn.Conv2d(in_channels=7, out_channels=kernel_size, kernel_size=kernel_size, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=7, out_channels=kernel_size, kernel_size=kernel_size, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.pools(X)