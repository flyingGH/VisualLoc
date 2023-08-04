import torch 
import torchvision 
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import cv2
import random
import numpy as np
import torch.nn.functional as F
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint




class calcDataset(Dataset):
    
    def __init__(self, standard_dataset):
        self.ds = standard_dataset

        size = ((160, 120))

        # preprocess with no transformation
        self.preprocess1 = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        
        # preprocess with random perspective
        self.preprocess2 = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((int(size[0] * 1.7), int(size[1] * 1.7))),
            transforms.RandomPerspective(distortion_scale=0.4, p = 1.),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])


        winsize = (64, 64)
        blocksize = (32, 32)
        blockstride = (32, 32)
        cellsize = (16, 16)
        nbins = 9

        self.hog = cv2.HOGDescriptor(winsize, blocksize, blockstride, cellsize, nbins)

    def __len__(self):
        return self.ds.__len__()


    def __getitem__(self, idx):
        img = self.ds.__getitem__(idx)[0]

        # preprocess the pair of images
        imgs = [self.preprocess1(img), self.preprocess2(img)]

        # shuffle them so target and input selection are random
        random.shuffle(imgs)

        target = np.array((imgs[1]*255).type(torch.uint8))[0]
        target = torch.Tensor(self.hog.compute(target))

        return imgs[0], target[None, :]





class calcDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size


    def setup(self, stage=None):
        train_ds = torchvision.datasets.CIFAR10(root="/home/oliver/Documents/github/Datasets", train=True, download=True)
        self.train_dataset = calcDataset(train_ds)
        test_ds = torchvision.datasets.CIFAR10(root="/home/oliver/Documents/github/Datasets", train=False, download=True)
        self.val_dataset = calcDataset(test_ds)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)


    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)






class CALC(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_dim = (1, 160, 120)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5,5), stride=2, padding=4)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=1, padding=2)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 4, kernel_size=(3,3), stride=1, padding=0)
        self.fc1 = nn.Linear(936, 1064)
        self.fc2 = nn.Linear(1064, 2048)
        self.fc3 = nn.Linear(2048, 4032)

        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.norm2(x)

        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    
    def descriptors(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.norm2(x)

        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat[:, None, :], y)
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat[:, None, :], y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=9e-4, momentum=0.9, weight_decay=5e-4)





if __name__ == '__main__': 
    model = CALC()
    data = calcDataModule()
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                             save_top_k=1,
                                             mode="min",
                                             dirpath='weights/')
    logger = TensorBoardLogger("tb_logs", name="calc")
    trainer = pl.Trainer(accelerator="gpu", devices=1, 
                       logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=data)


    