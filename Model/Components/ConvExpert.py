import torch
from torch import nn as  nn
import torch.functional as F
# import timmù
class ConvExpert(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, use_residual=True, kernel_size = 3):
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
        self.hidden_channel = out_channel * 4
        self.final_act = nn.SiLU(inplace=True)

        # Define experts operation
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=self.hidden_channel,
                kernel_size=1,
                padding=1 // 2
            ),
            nn.GroupNorm(num_groups=min(8, self.hidden_channel), num_channels=self.hidden_channel),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(
                in_channels=self.hidden_channel,
                out_channels=self.hidden_channel,
                kernel_size=3,
                padding=3 // 2
            ),
            nn.GroupNorm(num_groups=min(8, self.hidden_channel), num_channels=self.hidden_channel),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(
                in_channels=self.hidden_channel,
                out_channels= out_channel,
                kernel_size=1,
                padding=1 // 2
            ),
            nn.GroupNorm(num_groups=min(8, out_channel), num_channels=out_channel),
        )

        if self.use_residual:
            if in_channel == out_channel:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding= 1 // 2, bias = False)
            # self.drop_path = DropPath(dropout) if dropout > 0 else nn.Identity() 
        else :
            self.drop_path = nn.Identity()      
    
    def forward(self, X):
        out = self.conv_block(X)

        if self.use_residual:
           out = out + self.skip(X)
        return self.final_act(out)