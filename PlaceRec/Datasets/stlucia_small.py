from .dropbox_utils import dropbox_download_file
import zipfile
import os
import numpy as np
from .base_dataset import BaseDataset
import torchvision
import torch
import glob
from PIL import Image
from .utils import ImageDataset, collate_fn
from torch.utils.data import DataLoader
from scipy.signal import convolve2d


package_directory = os.path.dirname(os.path.abspath(__file__))


class StLucia_small(BaseDataset):
    """
    This is an abstract class that serves as a template for implementing 
    visual place recognition datasets. 

    Attributes: 
        query_paths (np.ndarray): A vector of type string providing relative paths to the query images
        map_paths (np.ndarray): A vector of type string providing relative paths to the map images
        name (str): A Name of the dataset
    """

    def __init__(self):
        # check to see if dataset is downloaded 
        if not os.path.isdir(package_directory + "/raw_images/StLucia_small"):
            print("====> Downloading StLucia_small Dataset")
            # download dataset as zip file
            dropbox_download_file("/vpr_datasets/StLucia_small.zip", package_directory + "/raw_images/StLucia_small.zip")
            # unzip the dataset 
            with zipfile.ZipFile(package_directory + "/raw_images/StLucia_small.zip","r") as zip_ref:
                os.makedirs(package_directory + "/raw_images/StLucia_small")
                zip_ref.extractall(package_directory + "/raw_images/")


        # load images
        self.map_paths = np.array(sorted(glob.glob(package_directory + "/raw_images/StLucia_small/180809_1545/*.jpg")))
        self.query_paths = np.array(sorted(glob.glob(package_directory + "/raw_images/StLucia_small/100909_0845/*.jpg")))

        self.name = "gardenspointwalking"


    def query_images(self, partition: str) -> np.ndarray:
        """
        This function returns the query images from the relevant partition of the dataset. 
        The partitions are either "train", "val", "test" or "all"

        args:
            partition (str): determines which partition the datasets query images to return.
                             must bet either "train", "val", "test", or "all"

        Returns: 
            np.ndarray: The query images as a numpy array in [N, H, W, C] format with datatype uint8

        """
        size = len(self.query_paths)

        # get the required partition of the dataset
        if partition == "train": paths = self.query_paths[:int(size*0.6)]
        elif partition == "val": paths = self.query_paths[int(size*0.6):int(size*0.8)]
        elif partition == "test": paths = self.query_paths[int(size*0.8):]
        elif partition == "all": paths = self.query_paths
        else: raise Exception("Partition must be 'train', 'val' or 'all'")
            
        return np.array([np.array(Image.open(pth)) for pth in paths])


    def map_images(self):
        """
        This function returns the map images.

        args:
            None:

        Returns: 
            np.ndarray: The query images as a numpy array in [N, H, W, C] format with datatype uint8

        """
        return np.array([np.array(Image.open(pth)) for pth in self.map_paths])



    def query_images_loader(self, partition: str, batch_size: int = 16, shuffle: bool = False,
                            augmentation: torchvision.transforms.transforms.Compose = None, 
                            pin_memory: bool = False, 
                            num_workers: int = 0) -> torch.utils.data.DataLoader:

        """
        This function returns a torch dataloader that can be used for loading 
        the query images in batches. The dataloader will reutn batches of [batch_size, H, W, C]
        where the datatype is uint8

        Args:
            partition (str): determines which partition the datasets query images to return.
                             must bet either "train", "val", "test", or "all"
            batch_size (int): The batch size for the dataloader to return
            shuffle: (bool): True if you want the order of query images to be shuffles.
                             False will keep the images in sequence. 
            augmentation (torchvision.transforms.transforms.Compose): The image augmentations
                             to apply to the query images 
            pin_memory (bool): pinning memory from cpu to gpu if using gpu for inference
            num_workers (int): number of worker used for proceesing the images by the dataloader. 
        
        Returns: 
            torch.utils.data.DataLoader: a dataloader for the query image set.
        
        """

        size = len(self.query_paths)

        # get the required partition of the dataset
        if partition == "train": paths = self.query_paths[:int(size*0.6)]
        elif partition == "val": paths = self.query_paths[int(size*0.6):int(size*0.8)]
        elif partition == "test": paths = self.query_paths[int(size*0.8):]
        elif partition == "all": paths = self.query_paths
        else: raise Exception("Partition must be 'train', 'val' or 'all'")

        # build the dataloader
        dataset = ImageDataset(paths, augmentation=augmentation)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, 
                                pin_memory=pin_memory, num_workers=num_workers, collate_fn=collate_fn)
        return dataloader


    def map_images_loader(self, partition: str, batch_size: int = 16, shuffle: bool = False,
                            augmentation: torchvision.transforms.transforms.Compose = None, 
                            pin_memory: bool = False, 
                            num_workers: int = 0) -> torch.utils.data.DataLoader:

        """
        This function returns a torch dataloader that can be used for loading 
        the map images in batches. The dataloader will reutn batches of [batch_size, H, W, C]
        where the datatype is uint8

        Args:
            partition (str): determines which partition the datasets map images to return.
                             must bet either "train", "val", "test", or "all"
            batch_size (int): The batch size for the dataloader to return
            shuffle: (bool): True if you want the order of map images to be shuffles.
                             False will keep the images in sequence. 
            augmentation (torchvision.transforms.transforms.Compose): The image augmentations
                             to apply to the map images 
            pin_memory (bool): pinning memory from cpu to gpu if using gpu for inference
            num_workers (int): number of worker used for proceesing the images by the dataloader. 
        
        Returns: 
            torch.utils.data.DataLoader: a dataloader for the map image set.
        
        """
        # build the dataloader
        dataset = ImageDataset(self.map_paths, augmentation=augmentation)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size,
                                pin_memory=pin_memory, num_workers=num_workers, collate_fn=collate_fn)
        return dataloader


    def ground_truth(self, partition: str, gt_type: str) -> np.ndarray:
        """
          This function return sthe relevant ground truth matrix given the partition 
          and ground truth type. The return matrix "GT" is of type bool where 
          GT[i, j] is true when query image i was taken from the place depicted by map image
          j

        Args:
            partition (str): determines which partition the datasets map images to return.
                             must bet either "train", "val", "test", or "all"
            gt_type (str): either "hard" or "soft". see https://arxiv.org/abs/2303.03281 for an 
                           explanation of soft and hard ground truth in Visual Place Recognition.

        Returns: 
            np.ndarray: A matrix GT of boolean type where GT[i, j] is true when 
            query image i was taken from the place depicted by map image. Otherwise it is false.
        """
        size = len(self.query_paths)
        
        gt_data = np.load(package_directory + '/raw_images/StLucia_small/GT.npz')

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



if __name__ == '__main__':
    ds = StLucia_small()