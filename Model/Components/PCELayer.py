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
                use_residual=True,
                downsampling=downsampling
            )
            for _ in range(num_experts)
        ])
        self.merge_gn = nn.GroupNorm(num_groups=min(8, out_channel), num_channels=out_channel)
        self.router_gate = RouterGate(gate_channel, hidden_size, num_experts)

        self.patch_size = patch_size
        self.fourier_freq = fourie_freq