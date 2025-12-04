import torch

from torch import nn

from .ConvExpert import ConvExpert
from Model.Components.Router.RouterGate import RouterGate

class PCELayer(nn.Module):
    def __init__(self,
                inpt_channel,
                out_channel,
                num_experts,
                dropout,
                patch_size,
                fourie_freq,
                gate_channel,
                hidden_size,
                downsampling,
                ):
        super().__init__()
        self.experts = nn.ModuleList([
            ConvExpert(
                in_channel=inpt_channel,
                out_channel=out_channel,
                dropout=dropout,
                use_residual=True
            )
            for _ in range(num_experts)
        ])

        self.router_gate = RouterGate(gate_channel, hidden_size, num_experts)
        hidden_channel = out_channel * 4

        self.inverted_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=hidden_channel,
                kernel_size=1,
                padding= 1 // 2,
                bias=False
            ),
            nn.GroupNorm(num_groups=min(8, hidden_channel), num_channels=hidden_channel),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                in_channels=hidden_channel,
                out_channels=hidden_channel,
                kernel_size=3,
                padding= 3 // 2,
                groups=hidden_channel,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(8, hidden_channel), num_channels=hidden_channel),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                in_channels=hidden_channel,
                out_channels= out_channel,
                kernel_size=1,
                stride = 2 if downsampling else 1, 
                padding= 1 // 2,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(8, out_channel), num_channels=out_channel),
            nn.SiLU(inplace=True),
        )

        self.patch_size = patch_size
        self.fourier_freq = fourie_freq