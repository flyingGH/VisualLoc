import torch
import numpy as np
from torch.utils.data import Dataset 
from PIL import Image
import pathlib
import pandas as pd
import dropbox
from dropbox.exceptions import AuthError
from .db_key import DROPBOX_ACCESS_TOKEN


class ImageDataset(Dataset):
    def __init__(self, img_paths, preprocess=None):
        self.img_paths = img_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])

        if self.preprocess is not None:
            img = self.preprocess(img)

        return np.array(img)



def dropbox_connect():
    """Create a connection to Dropbox."""

    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))
    return dbx




def dropbox_download_file(dropbox_file_path, local_file_path):
    """Download a file from Dropbox to the local machine."""

    try:
        dbx = dropbox_connect()

        with open(local_file_path, 'wb') as f:
            metadata, result = dbx.files_download(path=dropbox_file_path)
            f.write(result.content)
    except Exception as e:
        print('Error downloading file from Dropbox: ' + str(e))


def collate_fn(batch):
    return np.array(batch).transpose(0, 2, 3, 1)

