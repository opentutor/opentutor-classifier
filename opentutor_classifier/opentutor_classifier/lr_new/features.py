#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import math
from os import environ
import re
from typing import List, Tuple

from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from scipy import spatial

from opentutor_classifier.stopwords import STOPWORDS
from text_to_num import alpha2digit

from opentutor_classifier.utils import prop_bool
from .constants import FEATURE_LENGTH_RATIO, FEATURE_REGEX_AGGREGATE_DISABLED


def feature_regex_aggregate_disabled() -> bool:
    return prop_bool(FEATURE_REGEX_AGGREGATE_DISABLED, environ)


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
        if keyword == "[NEG]" and number_of_negatives(words)[0] == 0:
            is_there = False
            break
        elif keyword != "[NEG]" and keyword not in words:
            is_there = False
            break
    if is_there:
        return 1
    else:
        return 0


def feature_length_ratio_enabled() -> bool:
    enabled = environ.get(FEATURE_LENGTH_RATIO, "")
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


def feature_number_alignment(raw_example: str, raw_ideal: str, clustering) -> float:
    ex = re.findall(
        "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",  # noqa W605
        alpha2digit(raw_example, "en"),
    )
    ia = re.findall(
        "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",  # noqa W605
        alpha2digit(raw_ideal, "en"),
    )
    return clustering.word_alignment_feature(ex, ia)


def _calculate_similarity(a: np.ndarray, b: np.ndarray) -> float:
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


def regex_match_ratio(str_example: str, regexes: List[str]) -> float:
    if len(regexes) == 0:
        return 0
    count = 0
    for r in regexes:
        if re.search(r, str_example):
            count += 1
    return float(count / len(regexes))


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
