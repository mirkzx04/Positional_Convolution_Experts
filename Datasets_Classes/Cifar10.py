from torch.utils.data import Dataset
from torchvision import transforms as trs
from PIL import Image

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

        self.transforms = trs.Compose([
            trs.RandomCrop(32, padding=4),
            trs.CenterCrop(32),
            trs.RandomHorizontalFlip(p=0.5),
            trs.RandomRotation(degrees=15),
            trs.ColorJitter(brightness=0.2, contrast=0.2),
            trs.ToTensor(),
            trs.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
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

        img = Image.fromarray(imgaes)
        if self.transforms:
            img = self.transforms(img)

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

        self.transforms = trs.Compose([
            trs.Resize((32, 32)),
            trs.CenterCrop(32),
            trs.ToTensor(),
            trs.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

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

        if self.transforms:
            img = self.transforms(imgaes)

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
        for idx in range(1, 6):
            batch_path = f'{self.path}data_batch_{str(idx)}'
            batch_data = self.read_batch_path(path=batch_path)
            
            imgs = batch_data[b'data'].reshape(-1, 3, 32, 32)
            imgs = imgs.transpose(0, 2, 3, 1)

            self.data_train_list.append(imgs)
            self.labels_train_list += batch_data[b'labels']

        self.data_train = np.vstack(self.data_train_list).astype(np.uint8)
        self.labels_train = np.array(self.labels_train_list, dtype=np.int64)

    def load_batch_val(self):
        """
        Load validation batch data
        """
        batch_path = f'{self.path}test_batch'
        batch_data = self.read_batch_path(path=batch_path)

        # Stessa procedura per la validazione
        imgs = batch_data[b'data'].reshape(-1, 3, 32, 32)
        imgs = imgs.transpose(0, 2, 3, 1) # [N, 32, 32, 3]

        self.data_validation = imgs.astype(np.uint8)
        self.labels_validation = np.array(batch_data[b'labels'], dtype=np.int64)

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
