#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import math
import re

from typing import List, Tuple, Dict

import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer

avg_feature_vec_dp: Dict[str, np.array] = dict()


def _avg_feature_vector(
    words: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    sentence = " ".join(words)
    if sentence in avg_feature_vec_dp:
        return avg_feature_vec_dp[sentence]

    avg_feature_vec_dp[sentence] = model.encode(sentence, show_progress_bar=False)
    return avg_feature_vec_dp[sentence]


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


def regex_match_ratio(str_example: str, regexes: List[str]) -> float:
    if len(regexes) == 0:
        return 0
    count = 0
    for r in regexes:
        if re.search(r, str_example):
            count += 1
    return float(count / len(regexes))


def word2vec_example_similarity(
    model: SentenceTransformer,
    example: List[str],
    ideal: List[str],
) -> float:
    example_feature_vec = _avg_feature_vector(example, model=model)
    ia_feature_vec = _avg_feature_vector(ideal, model=model)
    return _calculate_similarity(example_feature_vec, ia_feature_vec)


def word2vec_question_similarity(model: SentenceTransformer, example, question):
    example_feature_vec = _avg_feature_vector(example, model=model)
    question_feature_vec = _avg_feature_vector(question, model=model)
    return _calculate_similarity(example_feature_vec, question_feature_vec)
