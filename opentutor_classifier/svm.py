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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List
import math
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import LeaveOneOut
import yaml
import re
from glob import glob
import pandas as pd
import json


from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ExpectationClassifierResult,
    load_data,
    load_yaml,
    InstanceConfigDefault,
    InstanceDefaultExpectationFeatures,
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
    ideal_answer: List[str]
    good_regex: List[str]
    bad_regex: List[str]


@dataclass
class InstanceConfig:
    question: str
    expectation_features: List[InstanceExpectationFeatures]

    def __post_init__(self):
        self.expectation_features = [
            x
            if isinstance(x, InstanceExpectationFeatures)
            else InstanceExpectationFeatures(**x)
            for x in self.expectation_features
        ]

    def write_to(self, file_path: str):
        with open(file_path, "w") as config_file:
            yaml.safe_dump(asdict(self), config_file)


@dataclass
class InstanceModels:
    models_by_expectation_num: Dict[int, svm.SVC]
    config: InstanceConfig


class SVMExpectationClassifier:
    def __init__(self):
        self.tag_map = defaultdict(lambda: wn.NOUN)
        self.tag_map["J"] = wn.ADJ
        self.tag_map["V"] = wn.VERB
        self.tag_map["R"] = wn.ADV
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
        data = [data]
        data = [entry.lower() for entry in data]
        data = [word_tokenize(entry) for entry in data]
        for entry in data:
            final_words = []
            for word, tag in pos_tag(entry):
                if word not in self.stopwords and word.isalpha():
                    final_words.append(word)
        return final_words

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
        return processed_data[0]

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
            question, model=word2vec, num_features=300, index2word_set=index2word_set
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

    def word_alignment_feature(self, example, ia, word2vec, index2word_set):
        cost = []
        n_exact_matches = len(set(ia).intersection(set(example)))
        ia, example = (
            list(set(ia).difference(example)),
            list(set(example).difference(ia)),
        )
        if not ia:
            return 1

        for ia_i in ia:
            inner_cost = []
            for e in example:
                dist = self.word2vec_example_similarity_feature(
                    word2vec, index2word_set, [e], [ia_i]
                )
                inner_cost.append(dist)
            cost.append(inner_cost)
        row_idx, col_idx = linear_sum_assignment(cost, maximize=True)
        score = (
            n_exact_matches + sum([cost[r][c] for r, c in zip(row_idx, col_idx)])
        ) / float(len(ia) + n_exact_matches)
        return score

    def length_ratio_feature(self, example, ideal_answer):
        return len(example) / len(ideal_answer)

    def number_of_negatives(self, example):
        negative_regex = r"\b(?:no|never|nothing|nowhere|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)\b"
        str_example = " ".join(example)
        replaced_example = re.sub("[.*'.*]", "", str_example)
        no_of_negatives = len(re.findall(negative_regex, replaced_example))
        if no_of_negatives % 2 == 0:
            even_negatives = 1
        else:
            even_negatives = 0
        return no_of_negatives, even_negatives

    def get_regex(self, exp_num, dict_expectation_features, regex_type):
        try:
            regex = dict_expectation_features[exp_num][regex_type]
        except Exception:
            regex = []

        return regex

    def good_regex_features(self, example, good_regex):
        str_example = " ".join(example)
        count = 0

        for r in good_regex:
            if re.search(r, str_example):
                count += 1
        try:
            return float(count / len(good_regex))
        except Exception:
            return 0

    def bad_regex_features(self, example, bad_regex):
        str_example = " ".join(example)
        count = 0
        for r in bad_regex:
            if re.search(r, str_example):
                count += 1
        try:
            return float(count / len(bad_regex))
        except Exception:
            return 0

    def calculate_features(
        self,
        question: List[str],
        example: List[str],
        ideal_answer: List[str],
        word2vec: Word2VecKeyedVectors,
        index2word_set: set,
        good_regex: List[str],
        bad_regex: List[str],
    ):
        feature_array = []

        good_regex_score = self.good_regex_features(example, good_regex)
        feature_array.append(good_regex_score)

        bad_regex_score = self.bad_regex_features(example, bad_regex)
        feature_array.append(bad_regex_score)

        no_of_negatives, even_negatives = self.number_of_negatives(example)
        feature_array.append(no_of_negatives)
        feature_array.append(even_negatives)

        feature_array.append(
            self.word_alignment_feature(example, ideal_answer, word2vec, index2word_set)
        )

        feature_array.append(self.length_ratio_feature(example, ideal_answer))

        feature_array.append(
            self.word2vec_example_similarity_feature(
                word2vec, index2word_set, example, ideal_answer
            )
        )
        feature_array.append(
            self.word2vec_question_similarity_feature(
                word2vec, index2word_set, example, question
            )
        )
        return feature_array

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

    def combine_dataset(self, data_root):
        training_data_list = [
            fn
            for fn in glob(path.join(data_root, "*/*.csv"))
            if not path.basename(path.dirname(fn)).startswith("default")
        ]
        config_list = glob(path.join(data_root, "*/*.yaml"))
        dataframes = []
        temp_data = {}

        for training_i, config_i in zip(training_data_list, config_list):
            loaded_df = load_data(training_i)
            loaded_config = load_yaml(config_i)
            temp_data["question"] = loaded_config["question"]
            exp_idx = loaded_df[
                loaded_df["exp_num"] != loaded_df["exp_num"].shift()
            ].index.tolist()
            loaded_df["exp_data"] = 0
            loaded_df["exp_num"] = 0
            for i in range(len(exp_idx) - 1):
                temp_data["ideal_answer"] = loaded_df.text[exp_idx[i]]
                r1 = exp_idx[i]
                r2 = exp_idx[i + 1]
                loaded_df["exp_data"][r1:r2] = json.dumps(temp_data)
            temp_data["ideal_answer"] = loaded_df.text[exp_idx[-1]]
            r3 = exp_idx[-1]
            r4 = len(loaded_df)
            loaded_df["exp_data"][r3:r4] = json.dumps(temp_data)
            dataframes.append(loaded_df)

        result = pd.concat(dataframes, axis=0)
        output_dir = path.join(data_root, "default")
        makedirs(output_dir, exist_ok=True)
        result.to_csv(path.join(output_dir, "training.csv"), index=False)
        return result

    def initialize_model(self):
        return svm.SVC(kernel="rbf", C=10, gamma="auto")

    def tune_hyper_parameters(self, model, parameters):
        model = GridSearchCV(
            model, parameters, cv=LeaveOneOut(), return_train_score=False
        )
        return model


@dataclass
class ExpectationToEvaluate:
    expectation: int
    classifier: svm.SVC


class SVMAnswerClassifierTraining:
    def __init__(self, shared_root: str = "shared"):
        self.word2vec = find_or_load_word2vec(path.join(shared_root, "word2vec.bin"))
        self.model_obj = SVMExpectationClassifier()
        self.model_instances: Dict[int, svm.SVC] = {}
        self.accuracy: Dict[int, int] = {}

    def default_train_all(
        self,
        data_root: str = "data",
        config_data: Dict = {},
        output_dir: str = "output",
    ) -> Dict:
        try:
            corpus = load_data(path.join(data_root, "default", "training.csv"))
        except Exception:
            corpus = self.model_obj.combine_dataset(data_root)
        model = self.model_obj.initialize_model()
        index2word_set = set(self.word2vec.index2word)
        output_dir = path.abspath(output_dir)
        makedirs(output_dir, exist_ok=True)

        def process_features(features, input_sentence, index2word_set):
            processed_input_sentence = self.model_obj.processing_single_sentence(
                input_sentence
            )
            processed_question = self.model_obj.processing_single_sentence(
                features["question"]
            )
            processed_ia = self.model_obj.processing_single_sentence(
                features["ideal_answer"]
            )

            features_list = self.model_obj.calculate_features(
                processed_question,
                processed_input_sentence,
                processed_ia,
                self.word2vec,
                index2word_set,
                [],
                [],
            )
            return features_list

        all_features = list(
            corpus.apply(
                lambda row: process_features(
                    json.loads(row["exp_data"]), row["text"], index2word_set
                ),
                axis=1,
            )
        )
        train_y = np.array(self.model_obj.encode_y(corpus["label"]))
        model.fit(all_features, train_y)

        results_loocv = model_selection.cross_val_score(
            model, all_features, train_y, cv=LeaveOneOut(), scoring="accuracy"
        )
        accuracy = results_loocv.mean() * 100.0
        self.model_instances[corpus["exp_num"].iloc[0]] = model
        self.model_obj.save(
            self.model_instances, path.join(output_dir, "models_by_expectation_num.pkl")
        )

        return accuracy

    def train_all(self, data_root: str = "data", output_dir: str = "output") -> Dict:
        config_path = path.join(data_root, "config.yaml")
        config = load_yaml(config_path)
        question = config.get("question")
        expectation_features = config.get("expectation_features") or []
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
                    ideal_answer=ia, good_regex=good_regex, bad_regex=bad_regex
                )
            )
            features = []
            for example in processed_data:
                feature = np.array(
                    self.model_obj.calculate_features(
                        processed_question,
                        example,
                        ia,
                        self.word2vec,
                        index2word_set,
                        good_regex,
                        bad_regex,
                    )
                )
                features.append(feature)
            train_y = np.array(self.model_obj.encode_y(train_y))

            model = self.model_obj.initialize_model()

            model.fit(features, train_y)
            results_loocv = model_selection.cross_val_score(
                model, features, train_y, cv=LeaveOneOut(), scoring="accuracy"
            )
            self.accuracy[exp_num] = results_loocv.mean() * 100.0
            self.model_instances[exp_num] = model
        self.model_obj.save(
            self.model_instances, path.join(output_dir, "models_by_expectation_num.pkl")
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

    def find_score_and_class(self, classifier, exp_num_i, sent_features):
        _evaluation = (
            "Good"
            if self.model_obj.predict(classifier, sent_features)[0] == 1
            else "Bad"
        )
        _score = self.model_obj.confidence_score(classifier, sent_features)

        return ExpectationClassifierResult(
            expectation=exp_num_i,
            evaluation=_evaluation,
            score=_score if _evaluation == "Good" else 1 - _score,
        )

    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        sent_proc = self.model_obj.processing_single_sentence(answer.input_sentence)
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

        if answer.config_data:
            conf = answer.config_data
            question_proc = self.model_obj.processing_single_sentence(conf.question)

            for i in range(len(conf.expectation_features_default)):

                ideal_answer = self.model_obj.processing_single_sentence(
                    conf.expectation_features_default[i].ideal_answer
                )
                sent_features = self.model_obj.calculate_features(
                    question_proc, sent_proc, ideal_answer, word2vec, index2word, [], []
                )
                exp_num = i
                classifier = expectations[0].classifier
                result.expectation_results.append(
                    self.find_score_and_class(classifier, exp_num, [sent_features])
                )
        else:
            conf2 = self.config()
            question_proc = self.model_obj.processing_single_sentence(conf2.question)

            for i in range(len(expectations)):
                sent_features = self.model_obj.calculate_features(
                    question_proc,
                    sent_proc,
                    conf2.expectation_features[i].ideal_answer,
                    word2vec,
                    index2word,
                    conf2.expectation_features[i].good_regex,
                    conf2.expectation_features[i].bad_regex,
                )
                exp_num = expectations[i].expectation
                classifier = expectations[exp_num].classifier
                result.expectation_results.append(
                    self.find_score_and_class(classifier, exp_num, [sent_features])
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


def train_default_classifier(
    data_root="data", config_data={}, shared_root="shared", output_dir: str = "out"
):
    svm_answer_classifier_training = SVMAnswerClassifierTraining(
        shared_root=shared_root
    )
    accuracy = svm_answer_classifier_training.default_train_all(
        data_root=data_root, config_data=config_data, output_dir=output_dir
    )
    return accuracy


def load_instances(
    model_root="./models",
    models_by_expectation_num_filename="models_by_expectation_num.pkl",
    config_filename="config.yaml",
) -> InstanceModels:
    try:
        with open(path.join(model_root, config_filename)) as config_file:
            config = InstanceConfig(**yaml.load(config_file, Loader=yaml.FullLoader))
    except Exception:
        config = InstanceConfig(question="", expectation_features=[])
    try:
        with open(
            path.join(model_root, models_by_expectation_num_filename), "rb"
        ) as models_file:
            models_by_expectation_num: Dict[int, svm.SVC] = pickle.load(models_file)
    except Exception:
        models_by_expectation_num = {}
    return InstanceModels(
        config=config, models_by_expectation_num=models_by_expectation_num
    )


def load_config_into_objects(config_data):
    if config_data:
        exp_feature_list = []
        for i in config_data["expectations"]:
            exp_feature_list.append(
                InstanceDefaultExpectationFeatures(ideal_answer=i["ideal"])
            )
        return InstanceConfigDefault(
            question=config_data["question"],
            expectation_features_default=exp_feature_list,
        )
