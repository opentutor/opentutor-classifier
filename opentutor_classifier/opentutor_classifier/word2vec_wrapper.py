
from typing import Any, Dict, List

from numpy import ndarray

from opentutor_classifier.word2vec import find_or_load_word2vec


class Word2VecWrapper:

    def __init__(self, path):
        self.model = find_or_load_word2vec(path)


    def get_feature_vectors(self, words: List[str] or set) -> Dict[str, ndarray]:
        result: List[ndarray] = []
        for word in words:
            if word in (self.model):
                result[word] = (self.model[word])
        return result

    def index_to_key(self) -> Any:
        return self.model.index_to_key