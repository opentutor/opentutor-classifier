#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
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
