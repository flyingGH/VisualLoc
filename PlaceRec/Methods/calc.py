import torch 
import torchvision 
from torchvision import transforms
import torch.nn as nn
import pytorch_lightning as pl

ds = torchvision.datasets.Places365(root="/Users/olivergrainge/Documents/github/Datasets", split="train-standard", small=True, 
                                    download=True)


"""


class CALC(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5,5), stride=2, padding=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 4, kernel_size=(3,3), stride=1, padding=0)
        self.fc1 = nn.Linear(4*14*19, 1064)
        self.fc2 = nn.Linear(1064, 2048)
        self.fc3 = nn.Linear(2048, 3648)

        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

    
    def descriptors(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.MSELoss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


"""