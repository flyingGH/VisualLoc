import numpy as np
from tqdm import tqdm 
from .base_method import BaseTechnique
from typing import Tuple
import torch
from torchvision import transforms
from torchvision.models import AlexNet_Weights
import sklearn


class AlexNet(BaseTechnique):
    
    def __init__(self):
        # Name of technique
        self.name = 'alexnet'

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.DEFAULT)
        self.model = self.model.features[:7]

        # send the model to relevant accelerator
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        # Dimensions to project into
        self.nDims = 4096

        # elimminate gradient computations and send to accelerator
        self.model.to(self.device)
        self.model.eval()

        # preprocess for network 
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 244], antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def compute_query_desc(self, query_images: np.ndarray) -> dict:
        desc = self.model(torch.Tensor(query_images.transpose(0, 3, 1, 2)).to(self.device)).detach().cpu().numpy()
        Ds = desc.reshape([query_images.shape[0], -1]) # flatten
        rng = np.random.default_rng(seed=0)
        Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')
        Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
        Ds = Ds @ Proj
        return {'query_descriptors': Ds}


    def compute_map_desc(self, map_images: np.ndarray) -> dict:
        desc = self.model(torch.Tensor(map_images.transpose(0, 3, 1, 2)).to(self.device)).detach().cpu().numpy()
        Ds = desc.reshape([map_images.shape[0], -1]) # flatten
        rng = np.random.default_rng(seed=0)
        Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')
        Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
        Ds = Ds @ Proj
        return {'map_descriptors': Ds}

    def set_map(self, map_descriptors: dict) -> None:
        try: 
            # try to implement with faiss
            import faiss
            self.map = faiss.IndexFlatIP(map_descriptors["map_descriptors"].shape[1])
            faiss.normalize_L2(map_descriptors["map_descriptors"])
            self.map.add(map_descriptors["map_descriptors"])

        except: 
            # faiss is not available on unix or windows systems. In this case 
            # implement with scikit-learn
            from sklearn.neighbors import NearestNeighbors
            self.map = NearestNeighbors(n_neighbors=10, algorithm='ball_tree', 
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
        return np.eye(10)


        