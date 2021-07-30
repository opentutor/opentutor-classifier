from abc import abstractmethod
from typing import List, Tuple
import numpy as np

from sklearn.cluster import KMeans, DBSCAN

from .features import (
    _avg_feature_vector
)

class OptimalCluster():
    @abstractmethod
    def getEmbedding(self, sentence: List[str]):
        pass

    @abstractmethod
    def getOptimalClusters(self, data: np.ndarray):
        pass

class OptimalClusterUsingDbScanWord2vec(OptimalCluster):
    def __init__(self, word2vec: None, index2word: None):
        self.word2vec = word2vec
        self.index2word = index2word
        self.data = None
    
    def getEmbedding(self, sentence: List[str]):
        return _avg_feature_vector( words=sentence, model=self.word2vec, num_features=300, index2word_set=self.index2word)

    def getOptimalClusters(self, data: np.ndarray):
        self.data = [ self.getEmbedding(example) for example in data ]

        dbscan = DBSCAN(eps=0.5)
        dbscan.fit(self.data)
        return len(np.unique(dbscan.labels_))


class OptimalClusterUsingKMeansWord2vec(OptimalCluster):
    def __init__(self, word2vec: None, index2word: None):
        self.word2vec = word2vec
        self.index2word = index2word
        self.data = None

    def fit_and_get_inertia(self, k: int) -> int:
        return KMeans(n_clusters=k).fit(self.data).inertia_

    def getEmbedding(self, sentence: List[str]):
        return _avg_feature_vector( words=sentence, model=self.word2vec, num_features=300, index2word_set=self.index2word)
        
    def getOptimalClusters(self, data: np.ndarray):
        self.data = [ self.getEmbedding(example) for example in data ]
        def distance_to_line(x1,y1, x2,y2, x0,y0):
            return abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))/((x2-x1)**2+(y2-y1)**2)**0.5

        # init_inertia = self.fit_and_get_inertia(1)

        # k = 2
        # new_inertia = self.fit_and_get_inertia(k)
        # while new_inertia < stopping_criterion * init_inertia:
        #     k+=1
        #     init_inertia, new_inertia = new_inertia, self.fit_and_get_inertia(k)

        # return k-1

        x1, y1 = 1, self.fit_and_get_inertia(1)
        x2, y2 = 6, self.fit_and_get_inertia(6)

        prev = 0
        for i in range(2,6):
            new = distance_to_line( x1, y1, x2, y2, i, KMeans(n_clusters=i).fit(self.data).inertia_)
            if new > prev: prev = new
            else: return i-1
        return 5

