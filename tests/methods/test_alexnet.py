import sys
sys.path.append('/Users/olivergrainge/Documents/github/VisualLoc')

from PlaceRec.Datasets import GardensPointWalking
from PlaceRec.Methods import AlexNet
import numpy as np
import unittest
import torch
from torchvision import transforms 


class alexnet_test(unittest.TestCase):

    def setUp(self):
        self.method = AlexNet()



class GenericMethodTest(alexnet_test):

    def test_name(self):
        assert isinstance(self.method.name, str)
        assert self.method.name.islower()


    def test_query_desc(self):
        imgs = np.random.rand(10, 255, 255, 3).astype(np.uint8)
        imgs = self.model.preprocess(t)




if __name__ == '__main__':
    unittest.main()