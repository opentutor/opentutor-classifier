#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict

import json
from os import path

from typing import Dict, List

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

from sklearn import model_selection, svm
from sklearn.model_selection import LeaveOneOut

from opentutor_classifier import (
    ARCH_SVM_CLASSIFIER,
    ArchLesson,
    DefaultModelSaveReq,
    AnswerClassifierTraining,
    ExpectationTrainingResult,
    ModelSaveReq,
    QuestionConfigSaveReq,
    TrainingConfig,
    TrainingInput,
    TrainingResult,
    ClassifierMode,
)
from .constants import FEATURE_REGEX_AGGREGATE
from .features import feature_regex_aggregate_enabled
from opentutor_classifier import DataDao
from opentutor_classifier.log import logger
from opentutor_classifier.stopwords import STOPWORDS
from .constants import MODEL_FILE_NAME
from .predict import (  # noqa: F401
    preprocess_sentence,
    SVMExpectationClassifier,
)
from opentutor_classifier.word2vec import find_or_load_word2vec


def _preprocess_trainx(data):
    pre_processed_dataset = []
    data = [entry.lower() for entry in data]
    data = [word_tokenize(entry) for entry in data]
    for entry in data:
        final_words = []
        for word, tag in pos_tag(entry):
            if word not in STOPWORDS and word.isalpha():
                final_words.append(word)
        pre_processed_dataset.append(final_words)
    return pre_processed_dataset


class SVMAnswerClassifierTraining(AnswerClassifierTraining):
    def __init__(self):
        self.model_obj = SVMExpectationClassifier()
        self.accuracy: Dict[int, int] = {}

    def configure(self, config: TrainingConfig) -> AnswerClassifierTraining:
        self.word2vec = find_or_load_word2vec(
            path.join(config.shared_root, "word2vec.bin")
        )
        return self

    def train_default(self, data: pd.DataFrame, dao: DataDao) -> TrainingResult:
        model = self.model_obj.initialize_model()
        index2word_set = set(self.word2vec.index_to_key)
        expectation_models: Dict[int, svm.SVC] = {}

        def process_features(features, input_sentence, index2word_set):
            processed_input_sentence = preprocess_sentence(input_sentence)
            processed_question = preprocess_sentence(features["question"])
            processed_ia = preprocess_sentence(features["ideal"])

            features_list = self.model_obj.calculate_features(
                processed_question,
                input_sentence,
                processed_input_sentence,
                processed_ia,
                self.word2vec,
                index2word_set,
                [],
                [],
                ClassifierMode.TRAIN,
            )
            return features_list

        all_features = list(
            data.apply(
                lambda row: process_features(
                    json.loads(row["exp_data"]), row["text"], index2word_set
                ),
                axis=1,
            )
        )
        train_y = np.array(self.model_obj.encode_y(data["label"]))
        model.fit(all_features, train_y)
        results_loocv = model_selection.cross_val_score(
            model, all_features, train_y, cv=LeaveOneOut(), scoring="accuracy"
        )
        accuracy = results_loocv.mean()
        expectation_models[data["exp_num"].iloc[0]] = model
        dao.save_default_pickle(
            DefaultModelSaveReq(
                arch=ARCH_SVM_CLASSIFIER,
                filename=MODEL_FILE_NAME,
                model=expectation_models,
            )
        )
        # need to write config for default even though it's empty
        # or will get errors later on attempt to load
        # QuestionConfig(question="").write_to(path.join(output_dir, "config.yaml"))
        return dao.create_default_training_result(
            ARCH_SVM_CLASSIFIER, ExpectationTrainingResult(accuracy=accuracy)
        )

    def train(self, train_input: TrainingInput, dao: DataDao) -> TrainingResult:
        question = train_input.config.question or ""
        if not question:
            raise ValueError("config must have a 'question'")
        train_data = (
            pd.DataFrame(
                [
                    [i, x.ideal, "good"]
                    for i, x in enumerate(train_input.config.expectations)
                    if x.ideal
                ],
                columns=["exp_num", "text", "label"],
            ).append(train_input.data, ignore_index=True)
        ).sort_values(by=["exp_num"], ignore_index=True)
        config_updated = train_input.config.clone()
        split_training_sets: dict = defaultdict(int)
        for i, exp_num in enumerate(train_data["exp_num"]):
            label = str(train_data["label"][i]).lower().strip()
            if label not in ["good", "bad"]:
                logger.warning(
                    f"ignoring training-data row {i} with invalid label {label}"
                )
                continue
            if exp_num not in split_training_sets:
                split_training_sets[exp_num] = [[], []]
            split_training_sets[exp_num][0].append(
                str(train_data["text"][i]).lower().strip()
            )
            split_training_sets[exp_num][1].append(label)
        index2word_set: set = set(self.word2vec.index_to_key)
        expectation_results: List[ExpectationTrainingResult] = []
        expectation_models: Dict[int, svm.SVC] = {}
        supergoodanswer = ""
        for exp_num in split_training_sets.keys():
            ideal = train_input.config.get_expectation_ideal(exp_num)
            if ideal:
                supergoodanswer = supergoodanswer + ideal
            else:
                supergoodanswer = supergoodanswer + split_training_sets[exp_num][0][0]
        for exp_num, (train_x, train_y) in split_training_sets.items():

            train_x.append(supergoodanswer)
            train_y.append("good")
            processed_data = _preprocess_trainx(train_x)
            processed_question = preprocess_sentence(question)
            ideal_answer = self.model_obj.initialize_ideal_answer(processed_data)
            good = train_input.config.get_expectation_feature(exp_num, "good", [])
            bad = train_input.config.get_expectation_feature(exp_num, "bad", [])
            config_updated.expectations[exp_num].features = {
                "good": good,
                "bad": bad,
                FEATURE_REGEX_AGGREGATE: feature_regex_aggregate_enabled(),
            }
            features = [
                np.array(
                    self.model_obj.calculate_features(
                        processed_question,
                        raw_example,
                        example,
                        ideal_answer,
                        self.word2vec,
                        index2word_set,
                        good,
                        bad,
                        mode=ClassifierMode.TRAIN,
                        expectation_config=train_input.config.expectations[exp_num]
                    )
                )
                for raw_example, example in zip(train_x, processed_data)
            ]
            import logging
            logging.warning(f"BEES{train_input.config.expectations[exp_num]}")
            train_y = np.array(self.model_obj.encode_y(train_y))
            model = self.model_obj.initialize_model()
            model.fit(features, train_y)
            results_loocv = model_selection.cross_val_score(
                model, features, train_y, cv=LeaveOneOut(), scoring="accuracy"
            )
            expectation_results.append(
                ExpectationTrainingResult(accuracy=results_loocv.mean())
            )
            expectation_models[exp_num] = model
        dao.save_config(
            QuestionConfigSaveReq(
                arch=ARCH_SVM_CLASSIFIER,
                lesson=train_input.lesson,
                config=config_updated,
            )
        )
        dao.save_pickle(
            ModelSaveReq(
                arch=ARCH_SVM_CLASSIFIER,
                lesson=train_input.lesson,
                filename="models_by_expectation_num.pkl",
                model=expectation_models,
            )
        )
        return TrainingResult(
            expectations=expectation_results,
            lesson=train_input.lesson,
            models=dao.get_model_root(
                ArchLesson(arch=ARCH_SVM_CLASSIFIER, lesson=train_input.lesson)
            ),
        )
