from typing import Any, Dict

from numpy import ndarray

from opentutor_classifier.word2vec import find_or_load_word2vec


class Word2VecWrapper:
    def __init__(self, path, slim_path):
        self.model = find_or_load_word2vec(path)
        self.model_slim = find_or_load_word2vec(slim_path)

    def get_feature_vectors(self, words, slim: bool = False) -> Dict[str, ndarray]:
        result: Dict[str, ndarray] = dict()
        for word in words:
            if not slim:
                if word in self.model:
                    result[word] = self.model[word]
            elif word in self.model_slim:
                result[word] = self.model_slim[word]
        return result

    def index_to_key(self, slim: bool = False) -> Any:
        if slim:
            return self.model_slim.index_to_key
        return self.model.index_to_key
