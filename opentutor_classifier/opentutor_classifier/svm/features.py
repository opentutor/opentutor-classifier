#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import math
from .constants import FEATURE_REGEX_AGGREGATE
import re

from typing import List, Tuple

from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from os import environ



def feature_regex_aggregate_enabled() -> bool:
    enabled = environ.get(FEATURE_REGEX_AGGREGATE, "")
    return enabled == "1" or enabled.lower() == "true"

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


def _calculate_similarity(a: float, b: float) -> float:
    similarity = 1 - spatial.distance.cosine(a, b)
    return similarity if not math.isnan(similarity) else 0


def length_ratio_feature(example: List[str], ideal: List[str]) -> float:
    return len(example) / float(len(ideal)) if len(ideal) > 0 else 0.0


def number_of_negatives(example) -> Tuple[float, float]:
    negative_regex = r"\b(?:no|never|nothing|nowhere|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)\b"
    str_example = " ".join(example)
    replaced_example = re.sub("[.*'.*]", "", str_example)
    no_of_negatives = len(re.findall(negative_regex, replaced_example))
    return (no_of_negatives, 1 if no_of_negatives % 2 == 0 else 0)


def regex_match(str_example: str, regexes: List[str]) -> List[bool]:
    if len(regexes) == 0:
        return []
    matches = []
    for r in regexes:
        if re.search(r, str_example):
            matches.append(1)
        else:
            matches.append(0)
    return matches


def regex_match_ratio(str_example: str, regexes: List[str]) -> float:
    if len(regexes) == 0:
        return 0
    count = 0
    for r in regexes:
        if re.search(r, str_example):
            count += 1
    return float(count / len(regexes))


def word_alignment_feature(
    example: List[str], ia: List[str], word2vec: Word2VecKeyedVectors, index2word_set
) -> float:
    cost = []
    n_exact_matches = len(set(ia).intersection(set(example)))
    ia, example = (list(set(ia).difference(example)), list(set(example).difference(ia)))
    if not ia:
        return 1

    for ia_i in ia:
        inner_cost = []
        for e in example:
            dist = word2vec_example_similarity(word2vec, index2word_set, [e], [ia_i])
            inner_cost.append(dist)
        cost.append(inner_cost)
    row_idx, col_idx = linear_sum_assignment(cost, maximize=True)
    return (
        n_exact_matches + sum([cost[r][c] for r, c in zip(row_idx, col_idx)])
    ) / float(len(ia) + n_exact_matches)


def word2vec_example_similarity(
    word2vec: Word2VecKeyedVectors,
    index2word_set: set,
    example: List[str],
    ideal: List[str],
) -> float:
    example_feature_vec = _avg_feature_vector(
        example, model=word2vec, num_features=300, index2word_set=index2word_set
    )
    ia_feature_vec = _avg_feature_vector(
        ideal, model=word2vec, num_features=300, index2word_set=index2word_set
    )
    return _calculate_similarity(example_feature_vec, ia_feature_vec)


def word2vec_question_similarity(
    word2vec: Word2VecKeyedVectors, index2word_set: set, example, question
):
    example_feature_vec = _avg_feature_vector(
        example, model=word2vec, num_features=300, index2word_set=index2word_set
    )
    question_feature_vec = _avg_feature_vector(
        question, model=word2vec, num_features=300, index2word_set=index2word_set
    )
    return _calculate_similarity(example_feature_vec, question_feature_vec)
