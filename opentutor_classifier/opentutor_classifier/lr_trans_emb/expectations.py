from collections import defaultdict
from typing import List
from os import environ

from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn import model_selection, linear_model
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

from text_to_num import alpha2digit

from opentutor_classifier import ClassifierMode, ExpectationConfig
from . import features

from opentutor_classifier.utils import prop_bool
from .clustering_features import CustomAgglomerativeClustering
from .constants import FEATURE_LENGTH_RATIO, FEATURE_REGEX_AGGREGATE_DISABLED


def feature_regex_aggregate_disabled() -> bool:
    return prop_bool(FEATURE_REGEX_AGGREGATE_DISABLED, environ)


def feature_length_ratio_enabled() -> bool:
    enabled = environ.get(FEATURE_LENGTH_RATIO, "")
    return enabled == "1" or enabled.lower() == "true"


class LRExpectationClassifier:
    def __init__(self):
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
        model: SentenceTransformer,
        feature_generator: features.FeatureGenerator,
        good: List[str],
        bad: List[str],
        clustering: CustomAgglomerativeClustering,
        mode: ClassifierMode,
        expectation_config: ExpectationConfig = None,
        patterns: List[str] = None,
    ) -> List[float]:
        raw_example = alpha2digit(raw_example, "en")
        regex_good = features.FeatureGenerator.regex_match(raw_example, good)
        regex_bad = features.FeatureGenerator.regex_match(raw_example, bad)
        feat = [
            *features.FeatureGenerator.number_of_negatives(example),
            clustering.word_alignment_feature(example, ideal),
            feature_generator.word2vec_example_similarity(model, example, ideal),
            feature_generator.word2vec_question_similarity(model, example, question),
        ]
        if mode == ClassifierMode.TRAIN:
            if feature_length_ratio_enabled():
                feat.append(
                    features.FeatureGenerator.length_ratio_feature(example, ideal)
                )
            if feature_regex_aggregate_disabled():
                feat = feat + regex_good + regex_bad
            else:
                feat.append(
                    features.FeatureGenerator.regex_match_ratio(raw_example, good)
                )
                feat.append(
                    features.FeatureGenerator.regex_match_ratio(raw_example, bad)
                )
        elif mode == ClassifierMode.PREDICT:
            if not expectation_config:
                raise Exception("predict mode must pass in ExpectationConfig")
            if prop_bool(FEATURE_LENGTH_RATIO, expectation_config.features):
                feat.append(
                    features.FeatureGenerator.length_ratio_feature(example, ideal)
                )
            if prop_bool(FEATURE_REGEX_AGGREGATE_DISABLED, expectation_config.features):
                feat = feat + regex_good + regex_bad
            else:
                feat.append(
                    features.FeatureGenerator.regex_match_ratio(raw_example, good)
                )
                feat.append(
                    features.FeatureGenerator.regex_match_ratio(raw_example, bad)
                )
        if patterns:
            for pattern in patterns:
                feat.append(features.check_is_pattern_match(raw_example, pattern))
        return feat

    @staticmethod
    def initialize_model() -> linear_model.LogisticRegression:
        return linear_model.LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear"
        )
