from .base_method import BaseTechnique
import torch
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import numpy as np
from typing import Tuple
import torch
import cv2


try:
    import faiss
except: 
    pass


####################### PARAMETERS #########################
magic_width = 512
magic_height = 512
cell_size = 16  # HOG cell-size
bin_size = 8  # HOG bin size
image_frames = 1  # 1 for grayscale, 3 for RGB
descriptor_depth = bin_size * 4 * image_frames  # x4 is here for block normalization due to nature of HOG
ET = 0.4  # Entropy threshold, vary between 0-1.

total_no_of_regions = int((magic_width / cell_size - 1) * (magic_width / cell_size - 1))

#############################################################

#################### GLOBAL VARIABLES ######################

d1d2dot_matrix = np.zeros([total_no_of_regions, total_no_of_regions], dtype=np.float64)
d1d2matches_maxpooled = np.zeros([total_no_of_regions], dtype=np.float64)
d1d2matches_regionallyweighted = np.zeros([total_no_of_regions], dtype=np.float64)

matched_local_pairs = []
ref_desc = []


############################################################


def largest_indices_thresholded(ary):
    good_list = np.where(ary >= ET)
    #    no_of_good_regions=len(good_list[0])


    return good_list


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""

    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


# @jit(nopython=False)
def conv_match_dotproduct(d1, d2, regional_gd, total_no_of_regions):  # Assumed aspect 1:1 here

    global d1d2dot_matrix
    global d1d2matches_maxpooled
    global d1d2matches_regionallyweighted
    global matched_local_pairs

    np.dot(d1, d2, out=d1d2dot_matrix)

    np.max(d1d2dot_matrix, axis=1, out=d1d2matches_maxpooled)  # Select best matched ref region for every query region

    np.multiply(d1d2matches_maxpooled, regional_gd,
                out=d1d2matches_regionallyweighted)  # Weighting regional matches with regional goodness

    score = np.sum(d1d2matches_regionallyweighted) / np.sum(regional_gd)  # compute final match score

    return score


# ================================================== HOG ===============================================================


class HOG(BaseTechnique):
    def __init__(self):
        self.device = "cpu"
        # preprocess images for method (None required)
        self.preprocess = transforms.ToTensor()
        # initialize the map to None
        self.map = None
        # method name
        self.name = "hog"

        # Method Parameters
        winSize = (512, 512)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (16, 16)
        nbins = 9

        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        self.winSize = winSize

    def compute_query_desc(self, query_images: np.ndarray) -> dict:
        ref_desc_list = []
        for ref_image in query_images:
            if ref_image is not None:
                img = ref_image * 255
                img = cv2.cvtColor(cv2.resize(img.astype(np.uint8), self.winSize), cv2.COLOR_BGR2GRAY)
                hog_desc = self.hog.compute(img)
            ref_desc_list.append(hog_desc)

        query_desc = np.array(ref_desc_list).astype(np.float32)
        return {"query_descriptors": query_desc}


    def compute_map_desc(self, map_images: np.ndarray) -> dict:
        ref_desc_list = []
        for ref_image in map_images:
            if ref_image is not None:
                img = ref_image * 255
                img = cv2.cvtColor(cv2.resize(img.astype(np.uint8), self.winSize), cv2.COLOR_BGR2GRAY)
                hog_desc = self.hog.compute(img)
            ref_desc_list.append(hog_desc)

        map_desc = np.array(ref_desc_list).astype(np.float32)
        return {"map_descriptors": map_desc}


    def set_map(self, map_descriptors: dict) -> None:
        try: 
            # try to implement with faiss
            self.map = faiss.IndexFlatIP(map_descriptors["map_descriptors"].shape[1])
            faiss.normalize_L2(map_descriptors["map_descriptors"])
            self.map.add(map_descriptors["map_descriptors"])

        except: 
            # faiss is not available on unix or windows systems. In this case 
            # implement with scikit-learn
            self.map = NearestNeighbors(n_neighbors=10, algorithm='auto', 
                                        metric='cosine').fit(map_descriptors["map_descriptors"])


    def place_recognise(self, query_images: np.ndarray, top_n: int=1) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.map, sklearn.neighbors._unsupervised.NearestNeighbors):
            desc = self.compute_query_desc(query_images)
            dist, idx = self.map.kneighbors(desc["query_descriptors"])
            return idx[:, :top_n], 1 - dist[:, :top_n]
        else: 
            desc = self.compute_query_desc(query_images)
            faiss.normalize_L2(desc["query_descriptors"])
            dist, idx = self.map.search(desc["query_descriptors"], top_n)
            return idx, dist


    def similarity_matrix(self, query_descriptors: dict, map_descriptors: dict) -> np.ndarray:
        return cosine_similarity(map_descriptors["map_descriptors"],
                                 query_descriptors["query_descriptors"]).astype(np.float32)