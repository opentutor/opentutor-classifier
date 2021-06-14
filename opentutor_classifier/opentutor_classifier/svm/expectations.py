from typing import List

from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn import model_selection, svm
from sklearn.preprocessing import LabelEncoder

from opentutor_classifier import ClassifierMode, ExpectationConfig

from opentutor_classifier.stopwords import STOPWORDS
from . import features
from opentutor_classifier.svm.constants import FEATURE_REGEX_AGGREGATE


def preprocess_sentence(sentence: str) -> List[str]:
    word_tokens_groups: List[str] = [
        word_tokenize(entry.lower())
        for entry in ([sentence] if isinstance(sentence, str) else sentence)
    ]
    result_words = []
    for entry in word_tokens_groups:
        for word, _ in pos_tag(entry):
            if word not in STOPWORDS and word.isalpha():
                result_words.append(word)
    return result_words


class SVMExpectationClassifier:
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
        mode: ClassifierMode,
        expectation_config: ExpectationConfig = None,
    ) -> List[float]:
        regex_good = features.regex_match(raw_example, good)
        regex_bad = features.regex_match(raw_example, bad)
        feat =  [
                *features.number_of_negatives(example),
                features.word_alignment_feature(
                    example, ideal, word2vec, index2word_set
                ),
                features.word2vec_example_similarity(
                    word2vec, index2word_set, example, ideal
                ),
                features.word2vec_question_similarity(
                    word2vec, index2word_set, example, question
                ),
            ]
        if mode == ClassifierMode.TRAIN:
            if features.feature_regex_aggregate_enabled():
                feat.append(regex_good + regex_bad)
            else:
                feat.append(features.regex_match_ratio(raw_example, good) + 
                    features.regex_match_ratio(raw_example, bad))
        elif mode == ClassifierMode.PREDICT:
            if not expectation_config:
                raise Exception("predict mode must pass in ExpectationConfig")
            if expectation_config.features[FEATURE_REGEX_AGGREGATE]:
                feat.append(feat.append(regex_good + regex_bad))
            else: 
                feat.append(features.regex_match_ratio(raw_example, good) + 
                    features.regex_match_ratio(raw_example, bad))
        return feat

    @staticmethod
    def initialize_model() -> svm.SVC:
        return svm.SVC(kernel="rbf", C=10, gamma="auto")
