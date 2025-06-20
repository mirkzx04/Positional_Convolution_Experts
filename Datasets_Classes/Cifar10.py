from torch.utils.data import Dataset

import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np

class CIFAR10TrainDataset(Dataset):
    def __init__(self, cifar10_dataset):
        """
        Constructor of CIFAR10TrainDataset

        Args :
            cifar10_dataset -> CIFAR10Dataset object

        path -> Dataset path
        data -> Images data
        lables -> Images labels
        """
        super().__init__()
        self.path = cifar10_dataset.path

        self.data = cifar10_dataset.data_train
        self.lables = cifar10_dataset.labels_train

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get item from dataset by index

        Args:
            idx -> Index of the item

        Returns:
            Tuple of image and label
        """
        imgaes = self.data[idx]
        labels = self.lables[idx]

        return imgaes, labels

class CIFAR10ValidationDataset(Dataset):
    def __init__(self, cifar10_dataset):
        """
        Constructor of CIFAR10ValidationDataset

        Args :
            cifar10_dataset -> CIFAR10Dataset object
        """
        super().__init__()
        self.path = cifar10_dataset.path

        self.data = cifar10_dataset.data_validation
        self.lables = cifar10_dataset.labels_validation

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get item from dataset by index

        Args:
            idx -> Index of the item

        Returns:
            Tuple of image and label
        """
        imgaes = self.data[idx]
        labels = self.lables[idx]

        return imgaes, labels

class CIFAR10Dataset(Dataset):
    def __init__(self, path):
        """
        Constructor of CIFAR10 dataset

        Args :
            Path -> dataset path
        """
        super().__init__()
        self.path = path

        self.data_train = []
        self.labels_train = []

        self.data_validation = []
        self.labels_validation = []

        # Load batch data from path
        self.load_batch_data()
        self.load_batch_val()

    def load_batch_data(self):
        for idx in range(1,6):
            batch_path = f'{self.path}{str(idx)}'
            batch_data = self.read_batch_path(path = batch_path)
            
            # Reshape IMG from [10000, 3072] to [10000, 3, 32,32]
            imgs_data = batch_data[b'data']
            imgs_data = imgs_data.reshape(10000, 3, 32,32)

            self.data_train.extend(imgs_data)
            self.labels_train.extend(batch_data[b'labels'])

    def load_batch_val(self):
        """
        Load validation batch data
        """
        batch_path = f'./Data/cifar-10-batches-py/test_batch'
        batch_data = self.read_batch_path(path = batch_path)

        # Reshape IMG from [10000, 3072] to [10000, 3, 32,32]
        imgs_data = batch_data[b'data']
        imgs_data = imgs_data.reshape(10000, 3, 32,32)
        
        self.data_validation.extend(imgs_data)
        self.labels_validation.extend(batch_data[b'labels'])

    # Unpickle function for cifar10 data
    def read_batch_path(self, path):
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, idx):
        imgaes = self.data_train[idx]
        labels = self.labels_train[idx]

        return imgaes, labels
