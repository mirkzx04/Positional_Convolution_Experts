from torch.utils.data import Dataset

import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np

class CIFAR10Dataset(Dataset):
    def __init__(self, path):
        """
        Constructor of CIFAR10Datase

        Args :
            Path -> dataset path

        path -> Dataset path
        data -> Images data
        lables -> Images labels
        """
        super().__init__()
        self.path = path
        self.data = []
        self.lables = []
        
        # Load batch data from path
        for idx in range(1,6):
            batch_path = f'{self.path}{str(idx)}'
            batch_data = self.load_batch(path = batch_path)
            
            # Reshape IMG from [10000, 3072] to [10000, 3, 32,32]
            imgs_data = batch_data[b'data']
            imgs_data = imgs_data.reshape(10000, 3, 32,32)

            # Upscale Image from 32x32 to 128x128
            upscale_img = self.upscale_img(imgs_data)
            self.data.extend(upscale_img)

            self.lables.extend(batch_data[b'labels'])

    # Unpickle function for cifar10 data
    def load_batch(self, path):
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        imgaes = self.data[idx]
        labels = self.lables[idx]

        return imgaes, labels
    
    def upscale_img(self, img, target_size = (128,128)):
        """
        Upscale dataset images
        """
        upscaled_img = []

        # Upscale image into batch
        for i, img in enumerate(img):
            # Convert shape from [3,32,32] to [32,32,3]
            img_hwc = img.transpose(1,2,0)

            upscaled = cv2.resize(
                img_hwc,
                target_size,
                interpolation=cv2.INTER_CUBIC
            )
            
            # Convert shape from [128,128,3] to [3,128,128]
            upscaled_img.append(upscaled.transpose(2,0,1))

        return np.array(upscaled_img)

    def show_images(self, num_img, idx):
        """
        Function for visualize some images
        Args:
            num_img -> Number of the images that want visualize
            idx -> Index of the images that want visualize, must be an array even of len 1
        """
        
        fig, axes = plt.subplots(2,4,figsize = (10,5))
        if idx is None:
            for i in range(num_img):
                img, lable = self.data[i], self.lables[i]
                print(img.shape)
                ax = axes[i//4, i %4]

                # Change channel from [C,H,W] to [H,W,C]
                ax.imshow(img.transpose(1,2,0))
                ax.set_title(f'Classes : {lable}')
                ax.axis('off')
            
            plt.show()
        else:
            for index in idx:
                img, lable = self.data[i], self.lables[i]
                ax = axes[i//4, i %4]

                # Change channel from [C,H,W] to [H,W,C]
                ax.imshow(img.permute(1,2,0))
                ax.set_title(f'Classes : {lable}')
                ax.axis('off')
            
            plt.show()