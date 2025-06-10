from torch import nn as  nn

class ConvExpert(nn.Module):
    def __init__(self, kernel_size, in_channel, out_channel, dropout, use_residual=True):
        super().__init__()
        """
        Constructor of one convolution expert
        It is a ResNet-like block with two convolution layers, batch normalization, ReLU activation

        Args :
            kernel_size -> kernel size of convolution
            in_channel -> input channel of convolution
            out_channel -> output channel of convolution
        """

        self.use_residual = use_residual
        self.in_channel = in_channel
        self.out_channel = out_channel

        # Define experts operation
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm2d(out_channel),
        )
        
        # Define residual connection
        self.skip_connection = None
        if in_channel != out_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm2d(out_channel)
            )

        # Final ReLU activation
        self.final_relu = nn.ReLU(inplace=True)
    def forward(self, X):

        # Setup identity for residual connection
        identity = X

        # Apply convolution block
        out = self.conv_block(X)

        # Skip connection
        if self.use_residual:
            if self.skip_connection is not None:
                identity = self.skip_connection(X)
            out += identity

        # Final ReLU activation
        out = self.final_relu(out)

        return out