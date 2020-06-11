from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from os import path, makedirs
import pickle
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict

from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ExpectationClassifierResult,
    loadData,
)


class SVMExpectationClassifier:
    def __init__(self):
        self.tag_map = defaultdict(lambda: wn.NOUN)
        self.tag_map["J"] = wn.ADJ
        self.tag_map["V"] = wn.VERB
        self.tag_map["R"] = wn.ADV
        self.ideal_answer = None
        self.model = None
        self.score_dictionary = defaultdict(int)
        np.random.seed(1)

    def preprocessing(self, data):
        preProcessedDataset = []
        data = [entry.lower() for entry in data]
        data = [word_tokenize(entry) for entry in data]
        for index, entry in enumerate(data):
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in stopwords.words("english") and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word, self.tag_map[tag[0]])
                    Final_words.append(word_Final)
            preProcessedDataset.append(Final_words)
        return preProcessedDataset

    def split(self, preProcessedDataset, target):
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
            preProcessedDataset, target, test_size=0.25
        )
        return Train_X, Test_X, Train_Y, Test_Y

    def initialize_ideal_answer(self, X):
        self.ideal_answer = X[0]
        return self.ideal_answer

    def encode_y(self, Train_Y, Test_Y):
        Encoder = LabelEncoder()
        Train_Y = Encoder.fit_transform(Train_Y)
        Test_Y = Encoder.fit_transform(Test_Y)
        return Train_Y, Test_Y

    def word_overlap_score(self, Train_X, ideal_answer):
        features = []
        for example in Train_X:
            intersection = set(ideal_answer).intersection(set(example))
            score = len(intersection) / len(set(ideal_answer))
            features.append(score)
        return features

    # function for extracting features
    def alignment(self, Train_X, ideal_answer):
        if ideal_answer is None:
            ideal_answer = self.ideal_answer
        features = self.word_overlap_score(Train_X, ideal_answer)
        return (np.array(features)).reshape(-1, 1)

    def get_params(self):
        C = 1.0
        kernel = "linear"
        degree = 3
        gamma = "auto"
        probability = True
        return C, kernel, degree, gamma, probability

    def set_params(self, **params):
        self.model = svm.SVC(
            C=params["C"],
            kernel=params["kernel"],
            degree=params["degree"],
            gamma=params["gamma"],
        )
        return self.model

    def train(self, trainFeatures, Train_Y):
        self.model.fit(trainFeatures, Train_Y)
        # print("Triaining complete")

    def predict(self, model, testFeatures):
        return model.predict(testFeatures)

    def find_accuracy(self, model_predictions, Test_Y):
        return accuracy_score(model_predictions, Test_Y) * 100

    def save(self, model_instances, filename):
        pickle.dump(model_instances, open(filename, "wb"))
        # print("Model saved successfully!")

    def confidence_score(self, model, sentence):
        return model.decision_function(sentence)[0]


@dataclass
class ExpectationToEvaluate:
    expectation: int
    classifier: svm.SVC


class SVMAnswerClassifierTraining:
    def __init__(self):
        self.model_obj = SVMExpectationClassifier()
        self.model_instances = {}
        self.ideal_answers_dictionary = {}
        self.accuracy = {}

    def train_all(self, corpus: pd.DataFrame, output_dir: str = ".") -> Dict:
        output_dir = path.abspath(output_dir)
        makedirs(output_dir, exist_ok=True)
        split_training_sets: dict = defaultdict(int)
        for i, value in enumerate(corpus["exp_num"]):
            if value not in split_training_sets:
                split_training_sets[value] = [[], []]
            split_training_sets[value][0].append(corpus["text"][i])
            split_training_sets[value][1].append(corpus["label"][i])
        for exp_num, (Train_X, Train_Y) in split_training_sets.items():
            processed_data = self.model_obj.preprocessing(Train_X)
            ia = self.model_obj.initialize_ideal_answer(processed_data)
            self.ideal_answers_dictionary[exp_num] = ia
            Train_X, Test_X, Train_Y, Test_Y = self.model_obj.split(
                processed_data, Train_Y
            )
            Train_Y, Test_Y = self.model_obj.encode_y(Train_Y, Test_Y)
            features = self.model_obj.alignment(Train_X, None)
            C, kernel, degree, gamma, probability = self.model_obj.get_params()
            model = self.model_obj.set_params(
                C=1.0, kernel="linear", degree=3, gamma="auto", probability=True
            )
            self.model_obj.train(features, Train_Y)
            self.model_instances[exp_num] = model

            training_predictions = self.model_obj.predict(model, features)
            self.accuracy[exp_num] = self.model_obj.find_accuracy(
                training_predictions, Train_Y
            )

        self.model_obj.save(
            self.model_instances, path.join(output_dir, "model_instances")
        )
        self.model_obj.save(
            self.ideal_answers_dictionary, path.join(output_dir, "ideal_answers")
        )
        return self.accuracy


class SVMAnswerClassifier:
    def __init__(self, model_instances, ideal_answers_dictionary):
        self.model_obj = SVMExpectationClassifier()
        self.model_instances = model_instances
        self.ideal_answers_dictionary = ideal_answers_dictionary

    def find_model_for_expectation(self, expectation: int) -> svm.SVC:
        return self.model_instances[expectation]

    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        sent_proc = self.model_obj.preprocessing(answer.input_sentence)
        expectations = (
            [
                ExpectationToEvaluate(
                    expectation=answer.expectation,
                    classifier=self.find_model_for_expectation(answer.expectation),
                )
            ]
            if answer.expectation != -1
            else [
                ExpectationToEvaluate(expectation=int(k), classifier=v)
                for k, v in self.model_instances.items()
            ]
        )
        result = AnswerClassifierResult(input=answer, expectationResults=[])
        for e in expectations:
            sent_features = self.model_obj.alignment(
                sent_proc, self.ideal_answers_dictionary[e.expectation]
            )
            result.expectationResults.append(
                ExpectationClassifierResult(
                    expectation=e.expectation,
                    evaluation=(
                        "Good"
                        if self.model_obj.predict(e.classifier, sent_features)[0] == 0
                        else "Bad"
                    ),
                    score=self.model_obj.confidence_score(e.classifier, sent_features),
                )
            )
        return result


def train_classifier(training_data_path: str, model_root: str = "."):
    training_data = loadData(training_data_path)
    svm_answer_classifier_training = SVMAnswerClassifierTraining()
    svm_answer_classifier_training.train_all(training_data, output_dir=model_root)


def load_instances(
    model_root="./models",
    model_filename="model_instances",
    ideal_answers_filename="ideal_answers",
) -> Tuple[dict, dict]:
    return (
        pickle.load(open(path.join(model_root, model_filename), "rb")),
        pickle.load(open(path.join(model_root, ideal_answers_filename), "rb")),
    )
