#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict
import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from os import path, makedirs
from sklearn import model_selection, svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List
import math
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import LeaveOneOut
import re
from glob import glob
import pandas as pd
import json


from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ExpectationClassifierResult,
    QuestionConfig,
    load_data,
    load_yaml,
)
from opentutor_classifier.speechact import SpeechActClassifier
from opentutor_classifier.stopwords import STOPWORDS
from .dtos import ExpectationToEvaluate, InstanceModels
from .utils import load_instances
from .word2vec import find_or_load_word2vec


def preprocess_sentence(sentence: str) -> List[str]:
    data: List[str] = [sentence]
    data = [entry.lower() for entry in data]
    data = [word_tokenize(entry) for entry in data]
    for entry in data:
        final_words = []
        for word, tag in pos_tag(entry):
            if word not in STOPWORDS and word.isalpha():
                final_words.append(word)
    return final_words


class SVMExpectationClassifier:
    def __init__(self):
        self.model = None
        self.score_dictionary = defaultdict(int)
        np.random.seed(1)

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

    def word_overlap_feature(self, example, ideal):
        intersection = set(ideal).intersection(set(example))
        score = len(intersection) / len(set(ideal))
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
        self, word2vec: Word2VecKeyedVectors, index2word_set, example, ideal
    ):
        example_feature_vec = self.avg_feature_vector(
            example, model=word2vec, num_features=300, index2word_set=index2word_set
        )
        ia_feature_vec = self.avg_feature_vector(
            ideal, model=word2vec, num_features=300, index2word_set=index2word_set
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

    def length_ratio_feature(self, example, ideal):
        return len(example) / len(ideal)

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
        ideal: List[str],
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
            self.word_alignment_feature(example, ideal, word2vec, index2word_set)
        )

        feature_array.append(self.length_ratio_feature(example, ideal))

        feature_array.append(
            self.word2vec_example_similarity_feature(
                word2vec, index2word_set, example, ideal
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
                temp_data["ideal"] = loaded_df.text[exp_idx[i]]
                r1 = exp_idx[i]
                r2 = exp_idx[i + 1]
                loaded_df["exp_data"][r1:r2] = json.dumps(temp_data)
            temp_data["ideal"] = loaded_df.text[exp_idx[-1]]
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


class SVMAnswerClassifier:
    def __init__(self, model_root="models", shared_root="shared"):
        self.model_root = model_root
        self.shared_root = shared_root
        self.model_obj = SVMExpectationClassifier()
        self._word2vec = None
        self._instance_models = None
        self.speech_act_classifier = SpeechActClassifier()

    def instance_models(self) -> InstanceModels:
        if not self._instance_models:
            self._instance_models = load_instances(model_root=self.model_root)
        return self._instance_models

    def models_by_expectation_num(self) -> Dict[int, svm.SVC]:
        return self.instance_models().models_by_expectation_num

    def config(self) -> QuestionConfig:
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
        sent_proc = preprocess_sentence(answer.input_sentence)
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

        result.speech_acts[
            "metacognitive"
        ] = self.speech_act_classifier.check_meta_cognitive(result)
        result.speech_acts["profanity"] = self.speech_act_classifier.check_profanity(
            result
        )
        print("resul = ", result)
        if answer.config_data:
            conf = answer.config_data
            question_proc = preprocess_sentence(conf.question)

            for i in range(len(conf.expectations)):

                ideal = preprocess_sentence(conf.expectations[i].ideal)
                sent_features = self.model_obj.calculate_features(
                    question_proc, sent_proc, ideal, word2vec, index2word, [], []
                )
                exp_num = i
                classifier = expectations[0].classifier
                result.expectation_results.append(
                    self.find_score_and_class(classifier, exp_num, [sent_features])
                )
        else:
            conf2 = self.config()
            question_proc = preprocess_sentence(conf2.question)

            for i in range(len(expectations)):
                sent_features = self.model_obj.calculate_features(
                    question_proc,
                    sent_proc,
                    conf2.expectations[i].ideal,
                    word2vec,
                    index2word,
                    conf2.expectations[i].good_regex,
                    conf2.expectations[i].bad_regex,
                )
                exp_num = expectations[i].expectation
                classifier = expectations[exp_num].classifier
                result.expectation_results.append(
                    self.find_score_and_class(classifier, exp_num, [sent_features])
                )
        return result
