import torch

from torch import nn

from .ConvExpert import ConvExpert

class PCELayer(nn.Module):
    def __init__(self,
                inpt_channel,
                out_channel,
                num_experts,
                dropout,
                patch_size,
                fourie_freq,
                gate_channel,
                ):
        super().__init__()
        self.experts = nn.ModuleList([
            ConvExpert(
                in_channel=inpt_channel,
                out_channel=out_channel,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

        flatten_channel = gate_channel * patch_size * patch_size
        self.router_gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_channel, num_experts)
        )

        self.patch_size = patch_size
        self.fourier_freq = fourie_freq