from torch.utils.data import Dataset
from torchvision import transforms as trs
from PIL import Image
import pickle
import numpy as np
import os

# --- CLASSE BASE (Master) ---
class CIFAR10Dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path

        self.data_train_list = []
        self.labels_train_list = []
        
        self.data_train = []
        self.labels_train = []
        self.data_validation = []
        self.labels_validation = []
        
        self.load_batch_data()
        self.load_batch_val()

    def load_batch_data(self):
        for idx in range(1, 6):
            file_name = f'data_batch_{idx}'
            batch_path = os.path.join(self.path, file_name)
            
            # Debug (opzionale)
            print(f"DEBUG: Caricamento {batch_path}")

            batch_data = self.read_batch_path(path=batch_path)
            
            # Reshape e Transpose per avere [H, W, C]
            imgs = batch_data[b'data'].reshape(-1, 3, 32, 32)
            imgs = imgs.transpose(0, 2, 3, 1)

            self.data_train_list.append(imgs)
            self.labels_train_list += batch_data[b'labels']

        # Concatenazione finale
        self.data_train = np.vstack(self.data_train_list).astype(np.uint8)
        self.labels_train = np.array(self.labels_train_list, dtype=np.int64)

    def load_batch_val(self):
        file_name = 'test_batch'
        batch_path = os.path.join(self.path, file_name)
        
        batch_data = self.read_batch_path(path=batch_path)

        imgs = batch_data[b'data'].reshape(-1, 3, 32, 32)
        imgs = imgs.transpose(0, 2, 3, 1) 

        self.data_validation = imgs.astype(np.uint8)
        self.labels_validation = np.array(batch_data[b'labels'], dtype=np.int64)

    def read_batch_path(self, path):
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, idx):
        return self.data_train[idx], self.labels_train[idx]


# --- TRAINING DATASET (Wrapper) ---
class CIFAR10TrainDataset(Dataset):
    def __init__(self, cifar10_dataset):
        super().__init__()
        self.path = cifar10_dataset.path
        self.data = cifar10_dataset.data_train
        self.lables = cifar10_dataset.labels_train

        self.transforms = trs.Compose([
            trs.RandomCrop(32, padding=4),
            trs.RandomHorizontalFlip(p=0.5),
            trs.RandomRotation(degrees=15),
            trs.ColorJitter(brightness=0.2, contrast=0.2),
            trs.ToTensor(),
            trs.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        imgaes = self.data[idx]
        labels = self.lables[idx]
        img = Image.fromarray(imgaes)
        
        if self.transforms:
            img = self.transforms(img)
        return img, labels


# --- VALIDATION DATASET (Wrapper) ---
class CIFAR10ValidationDataset(Dataset):
    def __init__(self, cifar10_dataset):
        super().__init__()
        self.path = cifar10_dataset.path
        self.data = cifar10_dataset.data_validation
        self.lables = cifar10_dataset.labels_validation

        self.transforms = trs.Compose([
            trs.ToTensor(),
            trs.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        imgaes = self.data[idx]
        labels = self.lables[idx]
        img = Image.fromarray(imgaes)

        if self.transforms:
            img = self.transforms(img)
        return img, labels