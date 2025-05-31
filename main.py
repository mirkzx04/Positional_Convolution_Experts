import torch

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from torch.optim import Adam

from Datasets_Classes.Cifar10 import CIFAR10Dataset
from Datasets_Classes.PatchExtractor import PatchExtractor
from Model.Components.Router import Router

if __name__ == "__main__":
    # Create CIFAR-10 dataset
    # cifar10 = CIFAR10Dataset('./Data/cifar-10-batches-py/data_batch_')

    # patch_extractor = PatchExtractor(16)

    # print(f'Shape of cifar 10 data : {cifar10.data.shape}')

    # cifar10_batch_patches = patch_extractor.extract_patches_coords(cifar10.data)
    
    # print(cifar10_batch_patches.shape)

    fake_patch = torch.rand(5, 64, 5, 128, 128)

    router = Router(4)

    router.initialize_keys(fake_patch)

