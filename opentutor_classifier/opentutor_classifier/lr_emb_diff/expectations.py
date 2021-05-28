from collections import defaultdict
import re
from typing import List

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn import model_selection, linear_model
from sklearn.preprocessing import LabelEncoder
from text_to_num import alpha2digit


from opentutor_classifier.stopwords import STOPWORDS
from . import features
from sentence_transformers import SentenceTransformer


word_mapper = {
    "n't": "not",
}


def preprocess_punctuations(sentence: str) -> str:
    sentence = re.sub(r'["\-"]', " - ", sentence)
    sentence = re.sub(r'["%"]', " percent ", sentence)
    return re.sub(r'["(", ")", "~", "!", "^", ",", "?", " ", "."]', " ", sentence)


def preprocess_sentence(sentence: str) -> List[str]:
    sentence = preprocess_punctuations(sentence)
    sentence = alpha2digit(sentence, "en")
    word_tokens_groups: List[str] = [
        word_tokenize(entry.lower())
        for entry in ([sentence] if isinstance(sentence, str) else sentence)
    ]
    result_words = []
    for entry in word_tokens_groups:
        for word, _ in pos_tag(entry):
            if word not in STOPWORDS:
                result_words.append(word)
    return [
        word_mapper.get(word, word)
        for word in result_words
        if len(word) != 1 or word.isnumeric()
    ]


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
        example: List[str],
        ideal: List[str],
        model: SentenceTransformer,
    ) -> List[float]:

        example_emb = model.encode(" ".join(example), show_progress_bar=False)
        ideal_emb = model.encode(" ".join(ideal), show_progress_bar=False)

        return list(example_emb - ideal_emb)

    @staticmethod
    def initialize_model() -> linear_model.LogisticRegression:
        return linear_model.LogisticRegression(tol=0.0001, C=1.0)
