from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from os import path, makedirs
import pickle
from sklearn import model_selection, svm
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List
import math
from scipy import spatial
from sklearn.model_selection import LeaveOneOut
import yaml
import re

from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ExpectationClassifierResult,
    load_data,
    load_yaml,
)

WORD2VEC_MODELS: Dict[str, Word2VecKeyedVectors] = {}


def find_or_load_word2vec(file_path: str) -> Word2VecKeyedVectors:
    abs_path = path.abspath(file_path)
    if abs_path not in WORD2VEC_MODELS:
        WORD2VEC_MODELS[abs_path] = KeyedVectors.load_word2vec_format(
            abs_path, binary=True
        )
    return WORD2VEC_MODELS[abs_path]


@dataclass
class InstanceExpectationFeatures:
    good_regex: List[str]
    bad_regex: List[str]
    expectation: int


@dataclass
class InstanceConfig:

    question: str
    expectation_features: List[InstanceExpectationFeatures]

    def write_to(self, file_path: str):
        with open(file_path, "w") as config_file:
            yaml.safe_dump(asdict(self), config_file)


@dataclass
class InstanceModels:
    models_by_expectation_num: Dict[int, svm.SVC]
    ideal_answers_by_expectation_num: Dict[int, List[str]]
    config: InstanceConfig


class SVMExpectationClassifier:
    def __init__(self):
        self.tag_map = defaultdict(lambda: wn.NOUN)
        self.tag_map["J"] = wn.ADJ
        self.tag_map["V"] = wn.VERB
        self.tag_map["R"] = wn.ADV
        self.ideal_answer = ""
        self.model = None
        self.score_dictionary = defaultdict(int)
        self.stopwords = set(
            [
                "i",
                "me",
                "my",
                "myself",
                "we",
                "our",
                "ours",
                "ourselves",
                "you",
                "you're",
                "you've",
                "you'll",
                "you'd",
                "your",
                "yours",
                "yourself",
                "yourselves",
                "he",
                "him",
                "his",
                "himself",
                "she",
                "she's",
                "her",
                "hers",
                "herself",
                "it",
                "it's",
                "its",
                "itself",
                "they",
                "them",
                "their",
                "theirs",
                "themselves",
                "what",
                "which",
                "who",
                "whom",
                "this",
                "that",
                "that'll",
                "these",
                "those",
                "am",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "having",
                "did",
                "doing",
                "a",
                "an",
                "the",
                "and",
                "but",
                "if",
                "or",
                "because",
                "as",
                "until",
                "while",
                "of",
                "at",
                "by",
                "for",
                "with",
                "about",
                "against",
                "between",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "to",
                "from",
                "up",
                "down",
                "in",
                "out",
                "on",
                "off",
                "over",
                "under",
                "again",
                "further",
                "then",
                "once",
                "here",
                "there",
                "when",
                "where",
                "why",
                "how",
                "all",
                "any",
                "each",
                "few",
                "more",
                "most",
                "other",
                "some",
                "such",
                "only",
                "own",
                "so",
                "than",
                "too",
                "very",
                "s",
                "t",
                "can",
                "will",
                "just",
                "should",
                "should've",
                "now",
                "d",
                "ll",
                "m",
                "o",
                "re",
                "ve",
                "y",
            ]
        )
        np.random.seed(1)

    def processing_single_sentence(self, data):
        processed_sentence = []
        data = [data]
        data = [entry.lower() for entry in data]
        data = [word_tokenize(entry) for entry in data]
        for entry in data:
            final_words = []
            for word, tag in pos_tag(entry):
                if word not in self.stopwords and word.isalpha():
                    final_words.append(word)
            processed_sentence.append(final_words)
        return processed_sentence

    def preprocessing(self, data):
        pre_processed_dataset = []
        data = [entry.lower() for entry in data]
        data = [word_tokenize(entry) for entry in data]

        for entry in data:
            final_words = []

            for word, tag in pos_tag(entry):

                if word not in self.stopwords and word.isalpha():
                    final_words.append(word)
            pre_processed_dataset.append(final_words)
        return pre_processed_dataset

    def split(self, pre_processed_dataset, target):
        train_x, test_x, train_y, test_y = model_selection.train_test_split(
            pre_processed_dataset, target, test_size=0.0
        )
        return train_x, test_x, train_y, test_y

    def initialize_ideal_answer(self, processed_data):
        self.ideal_answer = processed_data[0]
        return self.ideal_answer

    def encode_y(self, train_y):
        encoder = LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        return train_y

    def word_overlap_feature(self, example, ideal_answer):
        intersection = set(ideal_answer).intersection(set(example))
        score = len(intersection) / len(set(ideal_answer))
        return score

    def avg_feature_vector(self, words, model, num_features, index2word_set):
        feature_vec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        common_words = set(words).intersection(index2word_set)
        for word in common_words:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec, model[word])
        if nwords > 0:
            feature_vec = np.divide(feature_vec, nwords)
        return feature_vec

    def calculate_similarity(self, a, b):
        similarity = 1 - spatial.distance.cosine(a, b)
        if math.isnan(similarity):
            similarity = 0
        return similarity

    def word2vec_question_similarity_feature(
        self, word2vec: Word2VecKeyedVectors, index2word_set, example, question
    ):
        example_feature_vec = self.avg_feature_vector(
            example, model=word2vec, num_features=300, index2word_set=index2word_set
        )
        question_feature_vec = self.avg_feature_vector(
            question[0], model=word2vec, num_features=300, index2word_set=index2word_set
        )
        similarity = self.calculate_similarity(
            example_feature_vec, question_feature_vec
        )
        return similarity

    def word2vec_example_similarity_feature(
        self, word2vec: Word2VecKeyedVectors, index2word_set, example, ideal_answer
    ):
        example_feature_vec = self.avg_feature_vector(
            example, model=word2vec, num_features=300, index2word_set=index2word_set
        )
        ia_feature_vec = self.avg_feature_vector(
            ideal_answer,
            model=word2vec,
            num_features=300,
            index2word_set=index2word_set,
        )
        similarity = self.calculate_similarity(example_feature_vec, ia_feature_vec)
        return similarity

    def length_ratio_feature(self, example, ideal_answer):
        return len(example) / len(ideal_answer)

    def number_of_negatives(self, example):
        negative_regex = r"\b(?:no|never|nothing|nowhere|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)\b"
        str_example = " ".join(example)
        replaced_example = re.sub("[.*'.*]", "", str_example)
        no_of_negatives = len(re.findall(negative_regex, replaced_example))
        if no_of_negatives % 2 == 0:
            even_negatives = True
        else:
            even_negatives = False
        return no_of_negatives, even_negatives

    def get_regex(self, exp_num, dict_expectation_features, regex_type):
        try:
            regex = dict_expectation_features[exp_num][regex_type]
        except ValueError:
            regex = []

        return regex

    def good_regex_features(self, example, good_regex):
        str_example = " ".join(example)
        count = 0
        for r in good_regex:
            if re.search(r, str_example):
                count += 1
        return float(count / len(good_regex))

    def bad_regex_features(self, example, bad_regex):
        str_example = " ".join(example)
        count = 0
        for r in bad_regex:
            if re.search(r, str_example):
                count += 1
        return float(count / len(bad_regex))

    def expectation_features_to_dict(self, expectation_features):
        dict_expectation_features = {}
        for i in expectation_features:
            temp = {}
            temp["good_regex"] = i["good_regex"]
            temp["bad_regex"] = i["bad_regex"]
            dict_expectation_features[i["expectation"]] = temp
        return dict_expectation_features

    def calculate_features(
        self,
        question: str,
        train_x: np.ndarray,
        ideal_answer: List[str],
        word2vec: Word2VecKeyedVectors,
        index2word_set: set,
        good_regex: List[str],
        bad_regex: List[str],
    ):
        if not ideal_answer:
            ideal_answer = self.ideal_answer
        all_features = []
        for example in train_x:
            feature = []
            good_regex_score = self.good_regex_features(example, good_regex)
            bad_regex_score = self.bad_regex_features(example, bad_regex)
            feature.append(good_regex_score)
            feature.append(bad_regex_score)
            feature.append(self.length_ratio_feature(example, ideal_answer))
            feature.append(
                self.word2vec_example_similarity_feature(
                    word2vec, index2word_set, example, ideal_answer
                )
            )
            feature.append(
                self.word2vec_question_similarity_feature(
                    word2vec, index2word_set, example, question
                )
            )
            all_features.append(feature)
        return all_features

    def train(self, model, train_features, train_y):
        model.fit(train_features, train_y)
        return model

    def predict(self, model, test_features):
        return model.predict(test_features)

    def save(self, model_instances, filename):
        pickle.dump(model_instances, open(filename, "wb"))

    def confidence_score(self, model, sentence):
        score = model.decision_function(sentence)[0]
        x = score + model.intercept_[0]
        sigmoid = 1 / (1 + math.exp(-3 * x))
        return sigmoid


@dataclass
class ExpectationToEvaluate:
    expectation: int
    classifier: svm.SVC


class SVMAnswerClassifierTraining:
    def __init__(self, shared_root: str = "shared"):
        self.word2vec = find_or_load_word2vec(path.join(shared_root, "word2vec.bin"))
        self.model_obj = SVMExpectationClassifier()
        self.model_instances: Dict[int, svm.SVC] = {}
        self.ideal_answers_dictionary: Dict[int, List[str]] = {}
        self.accuracy: Dict[int, int] = {}

    def train_all(self, data_root: str = "data", output_dir: str = "output") -> Dict:
        config_path = path.join(data_root, "config.yaml")
        config = load_yaml(config_path)
        question = config.get("question")
        expectation_features = config.get("expectation_features")

        if not question:
            raise ValueError(f"config.yaml must have a 'question' at {config_path}")
        corpus = load_data(path.join(data_root, "training.csv"))
        output_dir = path.abspath(output_dir)
        makedirs(output_dir, exist_ok=True)
        split_training_sets: dict = defaultdict(int)
        for i, value in enumerate(corpus["exp_num"]):
            if value not in split_training_sets:
                split_training_sets[value] = [[], []]
            split_training_sets[value][0].append(corpus["text"][i])
            split_training_sets[value][1].append(corpus["label"][i])
        index2word_set: set = set(self.word2vec.index2word)
        expectation_features_objects = []
        for exp_num, (train_x, train_y) in split_training_sets.items():
            processed_data = self.model_obj.preprocessing(train_x)
            processed_question = self.model_obj.processing_single_sentence(question)
            ia = self.model_obj.initialize_ideal_answer(processed_data)
            good_regex = self.model_obj.get_regex(
                exp_num, expectation_features, "good_regex"
            )
            bad_regex = self.model_obj.get_regex(
                exp_num, expectation_features, "bad_regex"
            )

            expectation_features_objects.append(
                InstanceExpectationFeatures(
                    good_regex=good_regex, bad_regex=bad_regex, expectation=int(exp_num)
                )
            )

            self.ideal_answers_dictionary[exp_num] = ia
            features = np.array(
                self.model_obj.calculate_features(
                    processed_question,
                    processed_data,
                    [],
                    self.word2vec,
                    index2word_set,
                    good_regex,
                    bad_regex,
                )
            )
            train_y = np.array(self.model_obj.encode_y(train_y))
            model = svm.SVC()
            model.fit(features, train_y)

            results_loocv = model_selection.cross_val_score(
                model, features, train_y, cv=LeaveOneOut(), scoring="accuracy"
            )
            self.accuracy[exp_num] = results_loocv.mean() * 100.0
            self.model_instances[exp_num] = model
        self.model_obj.save(
            self.model_instances, path.join(output_dir, "models_by_expectation_num.pkl")
        )
        self.model_obj.save(
            self.ideal_answers_dictionary,
            path.join(output_dir, "ideal_answers_by_expectation_num.pkl"),
        )

        InstanceConfig(
            question=question, expectation_features=expectation_features_objects
        ).write_to(path.join(output_dir, "config.yaml"))
        return self.accuracy


class SVMAnswerClassifier:
    def __init__(self, model_root="models", shared_root="shared"):
        self.model_root = model_root
        self.shared_root = shared_root
        self.model_obj = SVMExpectationClassifier()
        self._word2vec = None
        self._instance_models = None

    def instance_models(self) -> InstanceModels:
        if not self._instance_models:
            self._instance_models = load_instances(model_root=self.model_root)
        return self._instance_models

    def models_by_expectation_num(self) -> Dict[int, svm.SVC]:
        return self.instance_models().models_by_expectation_num

    def find_ideal_answer(self, expectation_num: int) -> List[str]:
        return self.instance_models().ideal_answers_by_expectation_num[expectation_num]

    def config(self) -> InstanceConfig:
        return self.instance_models().config

    def find_model_for_expectation(self, expectation: int) -> svm.SVC:
        return self.models_by_expectation_num()[expectation]

    def find_word2vec(self) -> Word2VecKeyedVectors:
        if not self._word2vec:
            self._word2vec = find_or_load_word2vec(
                path.join(self.shared_root, "word2vec.bin")
            )
        return self._word2vec

    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        sent_proc = self.model_obj.processing_single_sentence(answer.input_sentence)
        question_proc = self.model_obj.processing_single_sentence(
            self.config().question
        )
        expectation_features = self.config().expectation_features

        dict_expectation_features = self.model_obj.expectation_features_to_dict(
            expectation_features
        )

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
                for k, v in self.models_by_expectation_num().items()
            ]
        )
        result = AnswerClassifierResult(input=answer, expectation_results=[])
        word2vec = self.find_word2vec()
        index2word = set(word2vec.index2word)
        for e in expectations:
            sent_features = self.model_obj.calculate_features(
                question_proc,
                sent_proc,
                self.find_ideal_answer(e.expectation),
                word2vec,
                index2word,
                self.model_obj.get_regex(
                    e.expectation, dict_expectation_features, "good_regex"
                ),
                self.model_obj.get_regex(
                    e.expectation, dict_expectation_features, "bad_regex"
                ),
            )
            result.expectation_results.append(
                ExpectationClassifierResult(
                    expectation=e.expectation,
                    evaluation=(
                        "Good"
                        if self.model_obj.predict(e.classifier, sent_features)[0] == 1
                        else "Bad"
                    ),
                    score=self.model_obj.confidence_score(e.classifier, sent_features),
                )
            )
        return result


def train_classifier(data_root="data", shared_root="shared", output_dir: str = "out"):
    svm_answer_classifier_training = SVMAnswerClassifierTraining(
        shared_root=shared_root
    )
    accuracy = svm_answer_classifier_training.train_all(
        data_root=data_root, output_dir=output_dir
    )
    return accuracy


def load_instances(
    model_root="./models",
    models_by_expectation_num_filename="models_by_expectation_num.pkl",
    ideal_answers_by_expectation_num_filename="ideal_answers_by_expectation_num.pkl",
    config_filename="config.yaml",
) -> InstanceModels:
    with open(path.join(model_root, config_filename)) as config_file:
        config = InstanceConfig(**yaml.load(config_file, Loader=yaml.FullLoader))
    with open(
        path.join(model_root, models_by_expectation_num_filename), "rb"
    ) as models_file:
        models_by_expectation_num: Dict[int, svm.SVC] = pickle.load(models_file)
    with open(
        path.join(model_root, ideal_answers_by_expectation_num_filename), "rb"
    ) as ideal_file:
        ideal_answers_by_expectation_num: Dict[int, List[str]] = pickle.load(ideal_file)
    return InstanceModels(
        config=config,
        models_by_expectation_num=models_by_expectation_num,
        ideal_answers_by_expectation_num=ideal_answers_by_expectation_num,
    )
