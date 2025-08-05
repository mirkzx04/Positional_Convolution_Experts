import torch

from torch import nn

from .ConvExpert import ConvExpert
from .FinalConv import FinalConv
class PCELayer(nn.Module):
    def __init__(self,
                inpt_channel,
                out_channel,
                num_experts,
                dropout,
                patch_size,
                fourie_freq):
        super().__init__()
        self.experts = nn.ModuleList([
            ConvExpert(
                in_channel=inpt_channel,
                out_channel=out_channel,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
        # self.final_conv = FinalConv(out_channel, out_channel)
        self.final_conv = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            padding=1,
            kernel_size=1
        )

        self.threshold = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, requires_grad=True))
        self.patch_size = patch_size
        self.fourier_freq = fourie_freq