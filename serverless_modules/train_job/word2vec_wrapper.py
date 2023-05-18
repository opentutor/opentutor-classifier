#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from typing import Any, Dict
import json
from numpy import ndarray
from serverless_modules.train_job.api import sbert_word_to_vec, get_sbert_index_to_key
from serverless_modules.logger import get_logger

logger = get_logger("w2v_wrapper")


class Word2VecWrapper:
    def __init__(self):
        self.loaded_word_vectors: Dict[str, ndarray] = {}
        self.words_with_no_sbert_vector: set = set()

    def get_feature_vectors(self, words: set, slim: bool = False) -> Dict[str, ndarray]:
        """
        Fetches words feature vectors from sbert service and stores word, vector pairs in memory
        """
        if len(words) == 0:
            return {}
        res_words = {}
        for word in words.copy():
            if word in self.loaded_word_vectors:
                res_words[word] = self.loaded_word_vectors[word]
                words.remove(word)
            if word in self.words_with_no_sbert_vector:
                words.remove(word)

        if len(words) > 0:
            sbert_w2v_result = sbert_word_to_vec(list(words), slim)
            self.loaded_word_vectors = {**self.loaded_word_vectors, **sbert_w2v_result}

            words_with_no_sbert_vector = set(
                filter(lambda word: word not in sbert_w2v_result, words)
            )
            self.words_with_no_sbert_vector = self.words_with_no_sbert_vector.union(
                words_with_no_sbert_vector
            )

            res_words = {**res_words, **sbert_w2v_result}
        return res_words

    def index_to_key(self, slim: bool = False) -> Any:
        return get_sbert_index_to_key(slim)
