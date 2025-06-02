import torch.nn as nn

from Components.Router import Router
from Components.ConvExpert import ConvExpert
from ..Datasets_Classes.PatchExtractor import PatchExtractor

class PCENetwork(nn.Module):
    def __init__(self, 
                 kernel_sz_experts,
                 out_cha_experts,
                 num_experts,
                 out_cha_router,
                 layer_number
                 ):
        super().__init__()

        """
        Constructor of PCE Network

        Args : 
            kernel_sz_experts -> kernel size of experts
            out_cha_experts -> out channel for convolution in experts
            num_experts -> Number of experts per layer
            out_cha_router -> out channel for conv projection in router
            layer_number -> Numer of layers
        """

        
