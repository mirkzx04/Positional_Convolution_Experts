import torch
import torch.functional as F

from torch import nn as  nn
# import timmù
class ConvExpert(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, use_residual=True, downsampling = False):
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
        self.hidden_size = out_channel * 1
        self.final_act = nn.SiLU(inplace=True)

        stride = 2 if downsampling else 1

        # Define experts operation
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=self.hidden_size,
                kernel_size=3,
                padding=1,
                stride= stride,
                bias=False
            ),
            nn.GroupNorm(num_groups=min(8, self.hidden_size), num_channels=self.hidden_size),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=out_channel,
                kernel_size=1,
                padding=0,
                bias=False
            ),
            nn.GroupNorm(num_groups=min(8, out_channel), num_channels=out_channel),
        )

        # if self.use_residual:
        #     if stride != 1 or in_channel != out_channel:
        #         self.skip = nn.Sequential(
        #             nn.Conv2d(
        #                 in_channels=in_channel,
        #                 out_channels=out_channel,
        #                 kernel_size=3,
        #                 padding=1,
        #                 stride= stride,
        #                 bias=False
        #             ),
        #             nn.GroupNorm(num_groups=min(8, out_channel), num_channels=out_channel),
        #         ) 
        #     else : 
        #         self.skip = nn.Identity()
    def forward(self, X):
        out = self.conv_block(X)

        # if self.use_residual:
        # #    out = self.drop_path(out) + self.skip(X)
        #     out += self.skip(X)
        # else : 
        #     out = self.drop_path(out)
        return self.final_act(out)