import torch
import numpy as np
from torch.utils.data import Dataset 
from PIL import Image



class ImageDataset(Dataset):
    def __init__(self, img_paths, augmentation=None):
        self.img_paths = img_paths
        self.augmentation = augmentation

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]))

        if self.augmentation is not None:
            img = self.augmentation(img)

        return img