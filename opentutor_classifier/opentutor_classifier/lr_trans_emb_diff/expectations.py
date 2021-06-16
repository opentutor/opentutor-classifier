from collections import defaultdict
from typing import List

from sklearn import model_selection, linear_model
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer


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
        return linear_model.LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear"
        )
