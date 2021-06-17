from collections import defaultdict
from typing import List

from sklearn import model_selection, linear_model
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer


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
    def initialize_ideal_answer(processed_data, model: SentenceTransformer):
        return model.encode(" ".join(processed_data[0]), show_progress_bar=False)

    @staticmethod
    def encode_y(train_y):
        encoder = LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        return train_y

    @staticmethod
    def calculate_features(
        example: List[str],
        ideal_answer: List[float],
        model: SentenceTransformer,
    ) -> List[float]:

        example_emb = model.encode(" ".join(example), show_progress_bar=False)
        return list(example_emb - ideal_answer)

    @staticmethod
    def initialize_model() -> linear_model.LogisticRegression:
        return linear_model.LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear"
        )
