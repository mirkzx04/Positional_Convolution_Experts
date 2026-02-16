import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

class TinyImageNetDataset(Dataset):
    def __init__(self, path):
        """
        Constructor of TinyImageNet dataset
        Args:
            Path : dataset path
        """
        super().__init__()
        self.path = path

        self.data_train = []   # Conterrà i PERCORSI (stringhe), non le immagini
        self.labels_train = []
        
        self.data_validation = [] # Conterrà i PERCORSI (stringhe)
        self.labels_validation = []

        # 1. Creiamo la mappa delle classi
        self.class_to_idx = self.create_class_mapping()
        
        # 2. Carichiamo i percorsi (molto veloce e leggero)
        self.load_batch_data() # Train
        self.load_batch_val()  # Validation

    def create_class_mapping(self):
        train_path = os.path.join(self.path, 'train')
        class_folders = sorted([
            d for d in os.listdir(train_path)
            if os.path.isdir(os.path.join(train_path, d))
        ])
        return {class_name: idx for idx, class_name in enumerate(class_folders)}

    def load_batch_data(self):
        """ Loading training paths """
        train_path = os.path.join(self.path, 'train')

        for class_name in os.listdir(train_path):
            if class_name not in self.class_to_idx: continue # Skip file spuri

            class_path = os.path.join(train_path, class_name, 'images')
            class_idx = self.class_to_idx[class_name]
            
            for image_file in os.listdir(class_path):
                if not image_file.endswith('.JPEG'): continue

                image_path = os.path.join(class_path, image_file)
                
                # CORREZIONE: Usiamo append, salviamo solo il percorso
                self.data_train.append(image_path)
                self.labels_train.append(class_idx)
    
    def load_batch_val(self):
        """ Loading validation paths """
        val_path = os.path.join(self.path, 'val')
        val_images_path = os.path.join(val_path, 'images')
        val_annotations_path = os.path.join(val_path, 'val_annotations.txt')
        
        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                image_name = parts[0]
                class_name = parts[1]
                
                image_path = os.path.join(val_images_path, image_name)
                
                if class_name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_name]
                    # CORREZIONE: Usiamo append, salviamo solo il percorso
                    self.data_validation.append(image_path)
                    self.labels_validation.append(class_idx)

    def __len__(self):
        # Questo metodo serve poco qui, ma lo lasciamo per compatibilità
        return len(self.data_train) + len(self.data_validation)

class TinyImageNetTrainDataset(Dataset):
    def __init__(self, tiny_imagenet_dataset):
        super().__init__()
        # Copiamo i riferimenti alle liste create dal padre
        self.data = tiny_imagenet_dataset.data_train
        self.labels = tiny_imagenet_dataset.labels_train

        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std  = (0.229, 0.224, 0.225)

        # La tua augmentation esatta
        self.train_transforms = T.Compose([
            # Input: PIL Image
            T.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomGrayscale(p=0.10),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(), # Converte in Tensor [C, H, W]
            T.Normalize(imagenet_mean, imagenet_std),
            T.RandomErasing(
                p=0.25,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value="random"
            ),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 1. Recupera il percorso
        img_path = self.data[idx]
        label = self.labels[idx]

        # 2. Carica immagine (Lazy Loading) - Essenziale convertire in RGB
        img_pil = Image.open(img_path).convert('RGB')

        # 3. Applica trasformazioni
        img_tensor = self.train_transforms(img_pil)
        
        return img_tensor, label

class TinyImageNetValidationDataset(Dataset):
    def __init__(self, tiny_imagenet_dataset):
        super().__init__()
        self.data = tiny_imagenet_dataset.data_validation
        self.labels = tiny_imagenet_dataset.labels_validation

        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std  = (0.229, 0.224, 0.225)
        
        self.val_transforms = T.Compose([
            T.Resize(256, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(imagenet_mean, imagenet_std),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img_pil = Image.open(img_path).convert('RGB')

        img_tensor = self.val_transforms(img_pil)
        
        return img_tensor, label