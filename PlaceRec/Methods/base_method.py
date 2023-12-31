from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple



class BaseTechnique(ABC):
    """ 
    This is an abstract class that serves as a template for visual place recognition 
    technique implementations. All abstract methods must be implemented in each technique. 

    Attributes: 
        map (fiass.index or sklearn.neighbors.NearestNeighbors): this is a structure consisting of all descriptors computed with 
                           "compute_map_desc". It is search by "place_recognise" when
                           performing place recognition 
    """

    map = None

    @abstractmethod
    def compute_query_desc(self, query_images: np.ndarray) -> dict:
        """
        computes the image descriptors of queries and returns them as a numpy array

        Args:
            query_images (np.ndarray): Images as a numpy array in (H, W, C) representation with a uint8 data type

        Returns: 
            dict: a dict describing the images. The contents of which will be determined by the particular method used.
        """
        pass

    @abstractmethod
    def compute_map_desc(self, map_images: np.ndarray) -> dict:
        """
        computes the image descriptors of the map and returns them as a dictionary. The particular format
        of the dictionary will depend on the type of vpr technique used

        Args:
            map_images (np.ndarray): Images as a numpy array in (H, W, C) representation with a uint8 data type
        
 Returns: 
            dict: a dict describing the images. The contents of which will be determined by the particular method used.
        """
        pass

    @abstractmethod
    def set_map(self, map_descriptors: dict) -> None:
        """
        Sets the map attribute of this class with the map descriptors computed with "compute_map_desc". This 
        map is searched by "place_recognise" to perform place recognition.

        Args:
            map (dict): dict: a dict describing the images. The contents of which will 
                             be determined by the particular description method used.

        Returns:
            None:
        """
        pass 

    @abstractmethod
    def place_recognise(self, queries: np.ndarray, top_n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs place recognition by computing query image representations and using them to search the map attribute
        to find the relevant map

        Args:
            queries (np.ndarray): Images as a numpy array in (H, W, C) representation with a uint8 data type

            top_n (int): Determines the top N places to match and return. 

        Returns: 
            Tuple[np.ndarray, np.ndarray]: a tuple with the first np.ndarray being a array of [N, H, W, C]
            images that match the query place. the second np.ndarray is a matrix of similarities of size
            [N, top_n] measuring the cosine similarity between the query images and returned places

        """
        pass

    @abstractmethod
    def similarity_matrix(self, query_descriptors: np.ndarray, map_descriptors: np.ndarray) -> np.ndarray:
        """
        computes the similarity matrix using the cosine similarity metric. It returns 
        a numpy matrix M. where M[i, j] determines how similar query i is to map image j.

        Args:
            query_descriptors (dict): dict of query descriptors computed by "compute_query_desc"
            map_descriptors (dict): dict of query descriptors computed by "compute_map_desc"

        Returns: 
            np.ndarray: matrix M where M[i, j] measures cosine similarity between query image i and map image j
        """
        pass