from torch.utils.data import Dataset
from torchvision import transforms as trs
from PIL import Image
import pickle
import numpy as np
import os

class CIFAR100Dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path

       
        self.data_train = []
        self.labels_train = []
        self.data_validation = []
        self.labels_validation = []
        
        self.load_batch_data()
        self.load_batch_val()

    def load_batch_data(self):
        file_name = 'train' 
        batch_path = os.path.join(self.path, file_name)
        
        print(f"DEBUG: Caricamento {batch_path}") # Debug

        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"Non trovo il file: {batch_path}. Assicurati di aver scaricato la versione 'python' di CIFAR-100")

        batch_data = self.read_batch_path(path=batch_path)
        
        # Reshape: [N, 3, 32, 32] -> [N, 32, 32, 3]
        imgs = batch_data[b'data'].reshape(-1, 3, 32, 32)
        imgs = imgs.transpose(0, 2, 3, 1)

        self.data_train = imgs.astype(np.uint8)
        self.labels_train = np.array(batch_data[b'fine_labels'], dtype=np.int64)

    def load_batch_val(self):
        file_name = 'test'
        batch_path = os.path.join(self.path, file_name)
        
        batch_data = self.read_batch_path(path=batch_path)

        imgs = batch_data[b'data'].reshape(-1, 3, 32, 32)
        imgs = imgs.transpose(0, 2, 3, 1) 

        self.data_validation = imgs.astype(np.uint8)
        self.labels_validation = np.array(batch_data[b'fine_labels'], dtype=np.int64)

    def read_batch_path(self, path):
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, idx):
        return self.data_train[idx], self.labels_train[idx]
    
# --- TRAINING DATASET (Wrapper) ---
class CIFAR100TrainDataset(Dataset):
    def __init__(self, cifar100_dataset):
        super().__init__()
        self.path = cifar100_dataset.path
        self.data = cifar100_dataset.data_train
        self.lables = cifar100_dataset.labels_train

        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)

        self.transforms = trs.Compose([
            trs.RandomCrop(32, padding=4, padding_mode="reflect"),
            trs.RandomHorizontalFlip(p=0.3),
            trs.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05
            ),
            trs.RandAugment(num_ops=2, magnitude=9),
            trs.ToTensor(),
            trs.RandomErasing(
                p=0.25,
                scale=(0.02, 0.20), 
                ratio=(0.30, 3.30),
                value="random"
            ),
            trs.Normalize(mean, std),
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
class CIFAR100ValidationDataset(Dataset):
    def __init__(self, cifar100_dataset):
        super().__init__()
        self.path = cifar100_dataset.path
        self.data = cifar100_dataset.data_validation
        self.lables = cifar100_dataset.labels_validation

        self.transforms = trs.Compose([
            trs.ToTensor(),
            trs.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
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