import torch

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from torch.optim import Adam

from Datasets_Classes.Cifar10 import CIFAR10Dataset

if __name__ == "__main__":
    # Load first batch from Data/cifar-10-batches-py

    cifar10 = CIFAR10Dataset('./Data/cifar-10-batches-py/data_batch_')

    print(f'Images len of the cifar-10 dataset : {cifar10.__len__()}')

    cifar10.show_images(5, None)