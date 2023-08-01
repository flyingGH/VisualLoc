from .base_method import BaseTechnique
import torch
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import numpy as np
from typing import Tuple


try:
    import faiss
except: 
    pass

class CosPlace(BaseTechnique):

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


        self.model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048, 
                                    verbose=False).to(self.device)
        self.model.eval()


        # preprocess images for method (None required)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # initialize the map to None
        self.map = None
        # method name
        self.name = "cosplace"


    def compute_query_desc(self, query_images: np.ndarray) -> dict:
        query_desc = self.model(torch.Tensor(query_images.transpose(0, 3, 1, 2)).to(self.device)).detach().cpu().numpy()
        return {"query_descriptors": query_desc}


    def compute_map_desc(self, map_images: np.ndarray) -> dict:
        map_desc = self.model(torch.Tensor(map_images.transpose(0, 3, 1, 2)).to(self.device)).detach().cpu().numpy()
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


        