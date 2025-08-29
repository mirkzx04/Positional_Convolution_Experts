import torch

from torch import nn

from .ConvExpert import ConvExpert
from .RouterGate import RouterGate

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

        self.router_gate = RouterGate(
            in_channel=gate_channel,
            hidden_size=hidden_size,
            num_experts=num_experts
        )

        self.patch_size = patch_size
        self.fourier_freq = fourie_freq