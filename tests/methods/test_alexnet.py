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
        self.ds = GardensPointWalking()



class GenericMethodTest(alexnet_test):

    def test_name(self):
        assert isinstance(self.method.name, str)
        assert self.method.name.islower()

    def test_query_desc(self):
        Q = self.ds.query_images("train", preprocess=self.method.preprocess)
        res = self.method.compute_query_desc(Q)
        assert isinstance(res, dict)
    
    def test_map_desc(self):
        Q = self.ds.map_images(preprocess=self.method.preprocess)
        res = self.method.compute_map_desc(Q)
        assert isinstance(res, dict)

    def test_map_loader(self):
        loader = self.ds.map_images_loader(preprocess=self.method.preprocess)
        for batch in loader:
            res = self.method.compute_query_desc(batch)
            assert isinstance(res, dict)
            break

    def test_query_loader(self):
        loader = self.ds.query_images_loader("test", preprocess=self.method.preprocess)
        for batch in loader:
            res = self.method.compute_query_desc(batch)
            assert isinstance(res, dict)
            break

    """ Next Test similarity matrix """

    """ Next Test set_map """

    """ Next Test place_recognise """




if __name__ == '__main__':
    unittest.main()