import zipfile
import os
import numpy as np
from .base_dataset import BaseDataset
import torchvision
import torch
from glob import glob
from PIL import Image
from .utils import ImageDataset, dropbox_download_file, collate_fn
from torch.utils.data import DataLoader


package_directory = os.path.dirname(os.path.abspath(__file__))



class SFU(BaseDataset):


    def __init__(self):
        # check to see if dataset is downloaded 
        if not os.path.isdir(package_directory + "/raw_images/SFU"):
            print("====> Downloading SFU Dataset")
            # download dataset as zip file
            dropbox_download_file("/vpr_datasets/SFU.zip", package_directory + "/raw_images/SFU.zip")
            # unzip the dataset 
            with zipfile.ZipFile(package_directory + "/raw_images/SFU.zip","r") as zip_ref:
                os.makedirs(package_directory + "/raw_images/SFU")
                zip_ref.extractall(package_directory + "/raw_images/")


        # load images
        self.map_paths = np.array(sorted(glob(package_directory + "/raw_images/SFU/dry/*.jpg")))
        self.query_paths = np.array(sorted(glob(package_directory + "/raw_images/SFU/jan/*.jpg")))

        self.name = "sfu"


    def query_images(self, partition: str, preprocess: torchvision.transforms.transforms.Compose = None ) -> np.ndarray:

        size = len(self.query_paths)

        # get the required partition of the dataset
        if partition == "train": paths = self.query_paths[:int(size*0.6)]
        elif partition == "val": paths = self.query_paths[int(size*0.6):int(size*0.8)]
        elif partition == "test": paths = self.query_paths[int(size*0.8):]
        elif partition == "all": paths = self.query_paths
        else: raise Exception("Partition must be 'train', 'val' or 'all'")
        
        if preprocess == None:
            return np.array([np.array(Image.open(pth)) for pth in paths])
        else: 
            imgs = np.array([np.array(Image.open(pth)) for pth in paths])
            return collate_fn(torch.stack([preprocess(q) for q in imgs]))



    def map_images(self, preprocess: torchvision.transforms.transforms.Compose = None):
                
        if preprocess == None:
            return np.array([np.array(Image.open(pth)) for pth in self.map_paths])
        else: 
            imgs = np.array([np.array(Image.open(pth)) for pth in self.map_paths])
            return collate_fn(torch.stack([preprocess(q) for q in imgs]))



    def query_images_loader(self, partition: str, batch_size: int = 16, shuffle: bool = False,
                            preprocess: torchvision.transforms.transforms.Compose = None, 
                            pin_memory: bool = False, 
                            num_workers: int = 0) -> torch.utils.data.DataLoader:

        size = len(self.query_paths)

        # get the required partition of the dataset
        if partition == "train": paths = self.query_paths[:int(size*0.6)]
        elif partition == "val": paths = self.query_paths[int(size*0.6):int(size*0.8)]
        elif partition == "test": paths = self.query_paths[int(size*0.8):]
        elif partition == "all": paths = self.query_paths
        else: raise Exception("Partition must be 'train', 'val' or 'all'")

        # build the dataloader
        dataset = ImageDataset(paths, preprocess=preprocess)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, 
                                pin_memory=pin_memory, num_workers=num_workers, collate_fn=collate_fn)
        return dataloader



    def map_images_loader(self, batch_size: int = 16, shuffle: bool = False,
                            preprocess: torchvision.transforms.transforms.Compose = None, 
                            pin_memory: bool = False, 
                            num_workers: int = 0) -> torch.utils.data.DataLoader:

        # build the dataloader
        dataset = ImageDataset(self.map_paths, preprocess=preprocess)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size,
                                pin_memory=pin_memory, num_workers=num_workers, collate_fn=collate_fn)
        return dataloader


    def ground_truth(self, partition: str, gt_type: str) -> np.ndarray:

        size = len(self.query_paths)
        
        gt_data = np.load(package_directory + '/raw_images/SFU/GT.npz')

        # load the full grount truth matrix with the relevant form
        if gt_type == "hard":
            gt = gt_data['GThard'].astype('bool')
        elif gt_type == "soft":
            gt = gt_data['GTsoft'].astype('bool')
        else: 
            raise Exception("gt_type must be either 'hard' or 'soft'")

        # select the relevant part of the ground truth matrix
        if partition == "train":
            gt = gt[:, :int(size*0.6)]
        elif partition == "val":
            gt = gt[:, int(size*0.6):int(size*0.8)]
        elif partition == "test":
            gt = gt[:, int(size*0.8):]
        elif partition == "all":
            pass
        else:
            raise Exception("partition must be either 'train', 'val', 'test' or 'all'")
        return gt.astype(bool)

