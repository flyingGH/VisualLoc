import sys
sys.path.append('/Users/olivergrainge/Documents/github/VisualLoc')

from PlaceRec.Datasets import GardensPointWalking
from PlaceRec.Methods import AlexNet
import numpy as np
import unittest
import torch
from torchvision import transforms 


class setup_test(unittest.TestCase):

    def setUp(self):
        self.method = AlexNet()
        self.ds = GardensPointWalking()
        self.sample_size = 10



class GenericMethodTest(setup_test):

    def test_name(self):
        assert isinstance(self.method.name, str)
        assert self.method.name.islower()

    def test_query_desc(self):
        Q = self.ds.query_images("train", preprocess=self.method.preprocess)[:self.sample_size]
        res = self.method.compute_query_desc(Q)
        assert isinstance(res, dict)
    
    def test_map_desc(self):
        Q = self.ds.map_images(preprocess=self.method.preprocess)[:self.sample_size]
        res = self.method.compute_map_desc(Q)
        assert isinstance(res, dict)

    def test_map_loader(self):
        loader = self.ds.map_images_loader(preprocess=self.method.preprocess, batch_size=self.sample_size)
        for batch in loader:
            res = self.method.compute_query_desc(batch)
            assert isinstance(res, dict)
            break

    def test_query_loader(self):
        loader = self.ds.query_images_loader("test", preprocess=self.method.preprocess, batch_size=self.sample_size)
        for batch in loader:
            res = self.method.compute_query_desc(batch)
            assert isinstance(res, dict)
            break

    def test_similarity_matrix(self):
        query_images = self.ds.query_images("test", preprocess=self.method.preprocess)[:self.sample_size]
        map_images = self.ds.map_images(preprocess=self.method.preprocess)[:self.sample_size]
        query_desc = self.method.compute_query_desc(query_images)
        map_desc = self.method.compute_map_desc(map_images)
        S = self.method.similarity_matrix(query_desc, map_desc)
        assert S.max() <= 1.
        assert S.min() >= 0.
        assert S.shape[0] == map_images.shape[0]
        assert S.shape[1] == query_images.shape[0]
        assert isinstance(S, np.ndarray)
        assert S.dtype == np.float32

    """ Next Test set_map """
    def test_set_map(self):
        map_images = self.ds.map_images(preprocess=self.method.preprocess)[:self.sample_size]
        map_desc = self.method.compute_map_desc(map_images)
        self.method.set_map(map_desc)
        assert self.method.map is not None


    """ Next Test place_recognise """
    def test_place_recognise(self):
        query_images = self.ds.query_images("test", preprocess=self.method.preprocess)[:self.sample_size]
        map_images = self.ds.map_images(preprocess=self.method.preprocess)[:self.sample_size]
        map_desc = self.method.compute_map_desc(map_images)
        self.method.set_map(map_desc)
        idx, score = self.method.place_recognise(query_images, top_n=3)
        assert isinstance(idx, np.ndarray)
        assert isinstance(score, np.ndarray)
        assert idx.shape[0] == query_images.shape[0]
        assert idx.shape[1] == 3
        assert score.shape[0] == query_images.shape[0]
        assert score.shape[1] == 3
        assert idx.dtype == int
        assert score.dtype == np.float32
        assert score.min() >= 0.
        assert score.max() <= 1.




if __name__ == '__main__':
    unittest.main()