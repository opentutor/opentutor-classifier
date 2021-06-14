from collections import defaultdict
import re
from typing import List

from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn import model_selection, linear_model
from sklearn.preprocessing import LabelEncoder
from text_to_num import alpha2digit

from opentutor_classifier import ClassifierMode, ExpectationConfig
from opentutor_classifier.stopwords import STOPWORDS
from . import features
from .clustering_features import CustomAgglomerativeClustering
from .constants import FEATURE_LENGTH_RATIO

word_mapper = {
    "n't": "not",
}


def preprocess_punctuations(sentence: str) -> str:
    sentence = re.sub(r"[\-]", " ", sentence)
    sentence = re.sub(r"[%]", " percent ", sentence)
    sentence = re.sub("n't", " not", sentence)
    sentence = re.sub(r"[()~!^,?.\'$=]", " ", sentence)
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
        if keyword == "[NEG]" and features.number_of_negatives(words)[0] == 0:
            is_there = False
            break
        elif keyword != "[NEG]" and keyword not in words:
            is_there = False
            break
    if is_there:
        return 1
    else:
        return 0


class LRExpectationClassifier:
    def __init__(self):
        self.model = None
        self.score_dictionary = defaultdict(int)

    @staticmethod
    def split(pre_processed_dataset, target):
        train_x, test_x, train_y, test_y = model_selection.train_test_split(
            pre_processed_dataset, target, test_size=0.0
        )
        return train_x, test_x, train_y, test_y

    @staticmethod
    def initialize_ideal_answer(processed_data):
        return processed_data[0]

    @staticmethod
    def encode_y(train_y):
        encoder = LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        return train_y

    @staticmethod
    def calculate_features(
        question: List[str],
        raw_example: str,
        example: List[str],
        ideal: List[str],
        word2vec: Word2VecKeyedVectors,
        index2word_set: set,
        good: List[str],
        bad: List[str],
        clustering: CustomAgglomerativeClustering,
        mode: ClassifierMode,
        expectation_config: ExpectationConfig = None,
        patterns: List[str] = None,
    ) -> List[float]:
        raw_example = alpha2digit(raw_example, "en")
        regex_good = features.regex_match(raw_example, good)
        regex_bad = features.regex_match(raw_example, bad)
        # feat = 
            # regex_good
            # + regex_bad+
            # [
        feat = [
                features.regex_match_ratio(raw_example, good),
                features.regex_match_ratio(raw_example, bad),
                *features.number_of_negatives(example),
                clustering.word_alignment_feature(example, ideal),
                features.length_ratio_feature(example, ideal),
                features.word2vec_example_similarity(
                    word2vec, index2word_set, example, ideal
                ),
                features.word2vec_question_similarity(
                    word2vec, index2word_set, example, question
                ),
            ]
        if mode == ClassifierMode.TRAIN:
            if features.feature_length_ratio_enabled():
                feat.append(features.length_ratio_feature(example, ideal))
        elif mode == ClassifierMode.PREDICT:
            if not expectation_config:
                raise Exception("predict mode must pass in ExpectationConfig")
            if expectation_config.features[FEATURE_LENGTH_RATIO]:
                feat.append(features.length_ratio_feature(example, ideal))
        if patterns:
            for pattern in patterns:
                feat.append(check_is_pattern_match(raw_example, pattern))
        return feat

    @staticmethod
    def initialize_model() -> linear_model.LogisticRegression:
        return linear_model.LogisticRegression(tol=0.0001, C=1.0)
