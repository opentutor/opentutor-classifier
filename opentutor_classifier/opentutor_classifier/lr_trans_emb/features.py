#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import math
import re
from typing import List, Tuple, Dict

from text_to_num import alpha2digit
from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from scipy import spatial
from sentence_transformers import SentenceTransformer

from opentutor_classifier.stopwords import STOPWORDS


word_mapper = {
    "n't": "not",
}


def preprocess_punctuations(sentence: str) -> str:
    sentence = re.sub(r"[\-=]", " ", sentence)
    sentence = re.sub(r"[%]", " percent ", sentence)
    sentence = re.sub("n't", " not", sentence)
    sentence = re.sub(r"[()~!^,?.\'$]", "", sentence)
    return sentence


def preprocess_sentence(sentence: str) -> List[str]:
    sentence = preprocess_punctuations(sentence.lower())
    sentence = alpha2digit(sentence, "en")
    word_tokens_groups: List[str] = [
        word_tokenize(entry)
        for entry in ([sentence] if isinstance(sentence, str) else sentence)
    ]
    result_words = []
    for entry in word_tokens_groups:
        for word, _ in pos_tag(entry):
            if word not in STOPWORDS:
                result_words.append(word)
    return [word_mapper.get(word, word) for word in result_words]


def check_is_pattern_match(sentence: str, pattern: str) -> int:
    words = preprocess_sentence(sentence)  # sentence should be preprocessed
    keywords = pattern.split("+")
    is_there = True
    for keyword in keywords:
        keyword = keyword.strip()
        if keyword == "[NEG]" and FeatureGenerator.number_of_negatives(words)[0] == 0:
            is_there = False
            break
        elif keyword != "[NEG]" and keyword not in words:
            is_there = False
            break
    if is_there:
        return 1
    else:
        return 0


class FeatureGenerator:
    def __init__(self):
        self.embedding_dp: Dict[Tuple[str, ...], np.ndarray] = dict()

    @staticmethod
    def _avg_feature_vector(
        words: List[str],
        model: Word2VecKeyedVectors,
        num_features: int,
        index2word_set: set,
    ) -> np.ndarray:
        feature_vec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        common_words = set(words).intersection(index2word_set)
        for word in common_words:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec, model[word])
        if nwords > 0:
            feature_vec = np.divide(feature_vec, nwords)
        return feature_vec

    @staticmethod
    def _calculate_similarity(a: float, b: float) -> float:
        similarity = 1 - spatial.distance.cosine(a, b)
        return similarity if not math.isnan(similarity) else 0

    @staticmethod
    def length_ratio_feature(example: List[str], ideal: List[str]) -> float:
        return len(example) / float(len(ideal)) if len(ideal) > 0 else 0.0

    @staticmethod
    def number_of_negatives(example) -> Tuple[float, float]:
        negative_regex = r"\b(?:no|never|nothing|nowhere|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)\b"
        str_example = " ".join(example)
        replaced_example = re.sub("[.*'.*]", "", str_example)
        no_of_negatives = len(re.findall(negative_regex, replaced_example))
        return (no_of_negatives, 1 if no_of_negatives % 2 == 0 else 0)

    @staticmethod
    def regex_match(str_example: str, regexes: List[str]) -> List[int]:
        if len(regexes) == 0:
            return []
        matches = []
        for r in regexes:
            if re.search(r, str_example):
                matches.append(1)
            else:
                matches.append(0)
        return matches

    @staticmethod
    def regex_match_ratio(str_example: str, regexes: List[str]) -> float:
        if len(regexes) == 0:
            return 0
        count = 0
        for r in regexes:
            if re.search(r, str_example):
                count += 1
        return float(count / len(regexes))

    def word2vec_example_similarity(
        self,
        model: SentenceTransformer,
        example: List[str],
        ideal: List[str],
    ) -> float:

        if tuple(example) not in self.embedding_dp:
            self.embedding_dp[tuple(example)] = model.encode(
                " ".join(example), show_progress_bar=False
            )

        if tuple(ideal) not in self.embedding_dp:
            self.embedding_dp[tuple(ideal)] = model.encode(
                " ".join(ideal), show_progress_bar=False
            )

        return FeatureGenerator._calculate_similarity(
            self.embedding_dp[tuple(example)], self.embedding_dp[tuple(ideal)]
        )

    def word2vec_question_similarity(
        self, model: SentenceTransformer, example, question
    ):
        if tuple(example) not in self.embedding_dp:
            self.embedding_dp[tuple(example)] = model.encode(
                " ".join(example), show_progress_bar=False
            )

        if tuple(question) not in self.embedding_dp:
            self.embedding_dp[tuple(question)] = model.encode(
                " ".join(question), show_progress_bar=False
            )

        return FeatureGenerator._calculate_similarity(
            self.embedding_dp[tuple(example)], self.embedding_dp[tuple(question)]
        )
