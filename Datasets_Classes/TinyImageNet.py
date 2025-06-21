import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset

class TinyImageNetTrainDataset(Dataset):
    def __init__(self, tiny_imagenet_dataset):
        """
        Constructor of TinyImageNetTrainDataset

        Args:
            tiny_imagenet_dataset -> TinyImageNetDataset object
        """
        super().__init__()
        self.path = tiny_imagenet_dataset.path
        self.data = tiny_imagenet_dataset.data_train
        self.labels = tiny_imagenet_dataset.labels_train

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
        images = self.data[idx]
        labels = self.labels[idx]
        return images, labels

class TinyImageNetValidationDataset(Dataset):
    def __init__(self, tiny_imagenet_dataset):
        """
        Constructor of TinyImageNetValidationDataset

        Args:
            tiny_imagenet_dataset -> TinyImageNetDataset object
        """
        super().__init__()
        self.path = tiny_imagenet_dataset.path
        self.data = tiny_imagenet_dataset.data_validation
        self.labels = tiny_imagenet_dataset.labels_validation

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
        images = self.data[idx]
        labels = self.labels[idx]
        return images, labels

class TinyImageNetDataset(Dataset):
    def __init__(self, path):
        """
        Constructor of TinyImageNet dataset

        Args:
            Path : dataset path
        """
        super().__init__()
        self.path = path

        self.data_train = []
        self.labels_train = []
        
        self.data_validation = []
        self.labels_validation = []

        # self.load_batch_data()
        # self.load_batch_val()

        self.class_to_idx = self.create_class_mapping()

    def create_class_mapping(self):
        """
        Create mapping from name to number indie 
        """
        train_path = os.path.join(self.path, 'train')
        class_folders  = sorted([
            d for d in os.listdir(train_path)
            if os.path.isdir(os.path.join(train_path, d))
        ])

        return {class_name : idx for idx, class_name in enumerate(class_folders)}

    def load_batch_data(self):
        """
        Loading training data
        """

        train_path = os.path.join(self.path, 'train')

        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name, 'images')

            class_idx = self.class_to_idx[class_name]
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                image = self.load_image(image_path)

                self.data_train.extend(image)
                self.data_validation.extend(class_idx)

        self.data_train = np.array(self.data_train)
        self.data_validation = np.array(self.data_validation)
    
    def load_batch_val(self):
        """ load validation data """
        val_path = os.path.join(self.path, 'val')
        val_images_path = os.path.join(val_path, 'images')
        val_annotations_path = os.path.join(val_path, 'val_annotations.txt')
        
        # Leggi annotations
        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                image_name = parts[0]
                class_name = parts[1]
                
                image_path = os.path.join(val_images_path, image_name)
                if class_name in self.class_to_idx:
                    image = self.load_image(image_path)
                    class_idx = self.class_to_idx[class_name]
                    
                    self.data_validation.append(image)
                    self.labels_validation.append(class_idx)

        self.data_validation = np.array(self.data_validation)
        self.labels_validation = np.array(self.labels_validation)

    def load_image(self, image_path, target_size=(64, 64)):
        """Carica e ridimensiona immagine"""
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        # Da [H, W, C] a [C, H, W]
        return image_array.transpose(2, 0, 1)
    

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, idx):
        images = self.data_train[idx]
        labels = self.labels_train[idx]
        return images, labels