import torch

from torch import nn

from .ConvExpert import ConvExpert
from Model.Components.Router.RouterGate import RouterGate

class PCELayer(nn.Module):
    def __init__(self,
                inpt_channel,
                out_channel,
                num_experts,
                patch_size,
                fourie_freq,
                gate_channel,
                downsampling,
                kernel_size,
                ):
        super().__init__()
        self.experts = nn.ModuleList([
            ConvExpert(
                in_channel=inpt_channel,
                out_channel=out_channel,
                use_residual=True,
                downsampling=downsampling,
                kernel_size = kernel_size
            )
            for _ in range(num_experts)
        ])
        self.merge_gn = nn.GroupNorm(num_groups=min(8, out_channel), num_channels=out_channel)
        self.activation_merge = nn.SiLU(inplace=True)
        self.router_gate = RouterGate(gate_channel, num_experts)
        self.gamma = nn.Parameter(torch.ones(1) * 1e-2)
        
        self.patch_size = patch_size
        self.fourier_freq = fourie_freq
        self.downsampling = downsampling

        self.post_block = nn.Sequential(
            nn.Conv2d(out_channels=out_channel, in_channels=out_channel, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_channels=out_channel, num_groups=min(8, out_channel)),
            nn.SiLU(inplace=True)
        )