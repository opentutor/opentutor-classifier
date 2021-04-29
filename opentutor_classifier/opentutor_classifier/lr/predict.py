#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict
from glob import glob
import json
from os import path, makedirs
from typing import Any, Dict, List, Optional, Tuple

from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import re
from sklearn import model_selection, linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from text_to_num import alpha2digit

from opentutor_classifier import (
    AnswerClassifier,
    AnswerClassifierInput,
    AnswerClassifierResult,
    ClassifierConfig,
    ExpectationClassifierResult,
    QuestionConfig,
)
from opentutor_classifier.utils import load_data, load_yaml
from opentutor_classifier.speechact import SpeechActClassifier
from opentutor_classifier.stopwords import STOPWORDS
from .dtos import ExpectationToEvaluate, InstanceModels

from . import features
from .utils import load_models
from opentutor_classifier.word2vec import find_or_load_word2vec


def _confidence_score(
    model: linear_model.LogisticRegression, sentence: List[List[float]]
) -> float:
    return model.predict_proba(sentence)[0, 1]


word_mapper = {
    "n't": "not",
}


def preprocess_punctuations(sentence: str) -> str:
    sentence = re.sub(r'["\-"]', " - ", sentence)
    sentence = re.sub(r'["%"]', " percent ", sentence)
    return re.sub(r'["(", ")", "~", "!", "^", ",", "?", " ", "."]', " ", sentence)


def preprocess_sentence(sentence: str) -> Tuple[Any, ...]:
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
    result_words = [
        word_mapper.get(word, word)
        for word in result_words
        if len(word) != 1 or word.isnumeric()
    ]
    return tuple(result_words)


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

    def calculate_features(
        self,
        question: List[str],
        raw_example: str,
        example: List[str],
        ideal: List[str],
        word2vec: Word2VecKeyedVectors,
        index2word_set: set,
        good: List[str],
        bad: List[str],
        patterns: List[str] = None,
    ) -> List[float]:
        raw_example = alpha2digit(raw_example, "en")
        feat = [
            features.regex_match_ratio(raw_example, good),
            features.regex_match_ratio(raw_example, bad),
            *features.number_of_negatives(example),
            features.word_alignment_feature(example, ideal, word2vec, index2word_set),
            features.length_ratio_feature(example, ideal),
            features.word2vec_example_similarity(
                word2vec, index2word_set, example, ideal
            ),
            features.word2vec_question_similarity(
                word2vec, index2word_set, example, question
            ),
        ]
        if patterns:
            for pattern in patterns:
                feat.append(check_is_pattern_match(raw_example, pattern))
        return feat

    def train(
        self,
        model: linear_model.LogisticRegression,
        train_features: np.ndarray,
        train_y: np.ndarray,
    ):
        model.fit(train_features, train_y)
        return model

    def predict(
        self, model: linear_model.LogisticRegression, test_features: np.ndarray
    ) -> np.ndarray:
        return model.predict(test_features)

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
        return linear_model.LogisticRegression(tol=0.0001, C=1.0)

    def tune_hyper_parameters(self, model, parameters):
        model = GridSearchCV(
            model, parameters, cv=LeaveOneOut(), return_train_score=False
        )
        return model


class LRAnswerClassifier(AnswerClassifier):
    def __init__(self):
        self.model_obj = LRExpectationClassifier()
        self._word2vec = None
        self._instance_models: Optional[InstanceModels] = None
        self.speech_act_classifier = SpeechActClassifier()

    def configure(
        self,
        config: ClassifierConfig,
    ) -> AnswerClassifier:
        self.model_name = config.model_name
        self.model_roots = config.model_roots
        self.shared_root = config.shared_root
        return self

    def instance_models(self) -> InstanceModels:
        if not self._instance_models:
            self._instance_models = load_models(
                self.model_name, model_roots=self.model_roots
            )
        return self._instance_models

    def models_by_expectation_num(self) -> Dict[int, linear_model.LogisticRegression]:
        return self.instance_models().models_by_expectation_num

    def config(self) -> QuestionConfig:
        return self.instance_models().config

    def find_model_for_expectation(
        self, expectation: int, return_first_model_if_only_one=False
    ) -> linear_model.LogisticRegression:
        m_by_e = self.models_by_expectation_num()
        return (
            m_by_e[0]
            if expectation >= len(m_by_e) and return_first_model_if_only_one
            else m_by_e[expectation]
        )

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
        _score = _confidence_score(classifier, sent_features)
        return ExpectationClassifierResult(
            expectation=exp_num_i,
            evaluation=_evaluation,
            score=_score if _evaluation == "Good" else 1 - _score,
        )

    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        sent_proc = preprocess_sentence(answer.input_sentence)
        conf = answer.config_data or self.config()
        expectations = [
            ExpectationToEvaluate(
                expectation=i,
                classifier=self.find_model_for_expectation(
                    i, return_first_model_if_only_one=True
                ),
            )
            for i in (
                [answer.expectation]
                if answer.expectation != -1
                else range(len(conf.expectations))
            )
        ]
        result = AnswerClassifierResult(input=answer, expectation_results=[])
        word2vec = self.find_word2vec()
        index2word = set(word2vec.index2word)
        result.speech_acts[
            "metacognitive"
        ] = self.speech_act_classifier.check_meta_cognitive(result)
        result.speech_acts["profanity"] = self.speech_act_classifier.check_profanity(
            result
        )
        question_proc = preprocess_sentence(conf.question)
        for exp in expectations:
            exp_conf = conf.expectations[exp.expectation]
            sent_features = self.model_obj.calculate_features(
                question_proc,
                answer.input_sentence,
                sent_proc,
                preprocess_sentence(exp_conf.ideal),
                word2vec,
                index2word,
                exp_conf.features.get("good") or [],
                exp_conf.features.get("bad") or [],
                exp_conf.features.get("patterns") or [],
            )
            result.expectation_results.append(
                self.find_score_and_class(
                    exp.classifier, exp.expectation, [sent_features]
                )
            )
        return result
