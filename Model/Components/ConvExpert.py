from torch import nn as  nn

class ConvExpert(nn.Module):
    def __init__(self, kernel_size, in_channel, out_channel, dropout):
        super().__init__()
        """
        Constructor of one convolution expert

        Args :
            kernel_size -> kernel size of convolution
            in_channel -> input channel of convolution
            out_channel -> output channel of convolution
        """

        # Define experts operation
        self.pools = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=4),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, padding=4),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, X):
        return self.pools(X)