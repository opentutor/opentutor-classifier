#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
from typing import Dict
from abc import ABC, abstractmethod

from numpy import ndarray, array
import openai
from openai.types.create_embedding_response import CreateEmbeddingResponse
from opentutor_classifier.constants import (
    DEPLOYMENT_MODE_OFFLINE,
)

from opentutor_classifier.openai.constants import OPENAI_API_KEY, OPENAI_ORG_ID_KEY
from opentutor_classifier.utils import require_env
from opentutor_classifier.word2vec import find_or_load_word2vec
from opentutor_classifier.api import sbert_word_to_vec

DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE") or DEPLOYMENT_MODE_OFFLINE

api_key = require_env(OPENAI_API_KEY, "DUMMY")
organization = require_env(OPENAI_ORG_ID_KEY, "DUMMY")

open_ai_client = openai.OpenAI(api_key=api_key)


class Word2VecWrapper(ABC):
    @abstractmethod
    def get_feature_vectors(self, words, slim: bool = False) -> Dict[str, ndarray]:
        raise NotImplementedError()


class Word2VecWrapperOffline(Word2VecWrapper):
    def __init__(self, path, slim_path):
        self.model = find_or_load_word2vec(path)
        self.model_slim = find_or_load_word2vec(slim_path)

    def get_feature_vectors(self, words, slim: bool = False) -> Dict[str, ndarray]:
        result: Dict[str, ndarray] = dict()
        word_list = list(words)
        if len(word_list) == 0:
            return result
        openai_result: CreateEmbeddingResponse = open_ai_client.embeddings.create(
            input=word_list, dimensions=300, model="text-embedding-3-small"
        )

        for idx, word in enumerate(word_list):
            embedding = array(openai_result.data[idx].embedding)
            result[word] = embedding

        # for word in words:
        #     if not slim:
        #         if word in self.model:
        #             result[word] = self.model[word]
        #     elif word in self.model_slim:
        #         result[word] = self.model_slim[word]
        return result


class Word2VecWrapperOnline(Word2VecWrapper):
    def __init__(self, path, slim_path):
        self.loaded_word_vectors: Dict[str, ndarray] = {}
        self.loaded_word_vectors_slim: Dict[str, ndarray] = {}
        self.words_with_no_sbert_vector: set = set()

    def get_feature_vectors(self, words: set, slim: bool = False) -> Dict[str, ndarray]:
        """
        Fetches words feature vectors from sbert service and stores word, vector pairs in memory
        """
        if len(words) == 0:
            return {}
        res_words = {}
        for word in words.copy():
            if slim is True and word in self.loaded_word_vectors_slim:
                res_words[word] = self.loaded_word_vectors_slim[word]
                words.remove(word)
            elif slim is False and word in self.loaded_word_vectors:
                res_words[word] = self.loaded_word_vectors[word]
                words.remove(word)
            elif word in self.words_with_no_sbert_vector:
                words.remove(word)

        if len(words) > 0:
            sbert_w2v_result = sbert_word_to_vec(list(words), slim)
            if slim:
                self.loaded_word_vectors_slim = {
                    **self.loaded_word_vectors_slim,
                    **sbert_w2v_result,
                }
            else:
                self.loaded_word_vectors = {
                    **self.loaded_word_vectors,
                    **sbert_w2v_result,
                }

            words_with_no_sbert_vector = set(
                filter(lambda word: word not in sbert_w2v_result, words)
            )
            self.words_with_no_sbert_vector = self.words_with_no_sbert_vector.union(
                words_with_no_sbert_vector
            )

            res_words = {**res_words, **sbert_w2v_result}
        return res_words


def get_word2vec(path, slim_path) -> Word2VecWrapper:
    return Word2VecWrapperOffline(path, slim_path)
