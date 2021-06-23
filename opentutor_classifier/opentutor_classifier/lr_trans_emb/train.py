#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict
import json
from os import path, environ

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model
from sklearn.model_selection import LeaveOneOut
from sentence_transformers import SentenceTransformer

from opentutor_classifier import DataDao
from opentutor_classifier import (
    ARCH_LR_TRANS_EMB_CLASSIFIER,
    PROP_TRAIN_QUALITY,
    AnswerClassifierTraining,
    ArchLesson,
    DefaultModelSaveReq,
    ExpectationTrainingResult,
    ModelSaveReq,
    QuestionConfigSaveReq,
    TrainingConfig,
    TrainingInput,
    TrainingResult,
    ClassifierMode,
)
from opentutor_classifier.config import get_train_quality_default
from opentutor_classifier.log import logger
from opentutor_classifier.utils import prop_bool


from .expectations import LRExpectationClassifier
from .features import preprocess_sentence, FeatureGenerator

from opentutor_classifier.sentence_transformer import find_or_load_sentence_transformer

from .clustering_features import CustomAgglomerativeClustering
from .constants import FEATURE_LENGTH_RATIO, FEATURE_REGEX_AGGREGATE_DISABLED


def _preprocess_trainx(data: List[str]) -> List[List[str]]:
    pre_processed_dataset = [preprocess_sentence(entry) for entry in data]
    return np.array(pre_processed_dataset)


def feature_regex_aggregate_disabled() -> bool:
    return prop_bool(FEATURE_REGEX_AGGREGATE_DISABLED, environ)


def feature_length_ratio_enabled() -> bool:
    enabled = environ.get(FEATURE_LENGTH_RATIO, "")
    return enabled == "1" or enabled.lower() == "true"


class LRAnswerClassifierTraining(AnswerClassifierTraining):
    def __init__(self):
        self.accuracy: Dict[int, int] = {}
        self.sentence_transformer: SentenceTransformer = None
        self.feature_generator = FeatureGenerator()
        self.train_quality = 1

    def configure(self, config: TrainingConfig) -> AnswerClassifierTraining:
        self.sentence_transformer = find_or_load_sentence_transformer(
            path.join(config.shared_root, "..", "sentence-transformer")
        )
        self.train_quality = config.properties.get(
            PROP_TRAIN_QUALITY, get_train_quality_default()
        )

        return self

    def train_default(self, data: pd.DataFrame, dao: DataDao) -> TrainingResult:
        model = LRExpectationClassifier.initialize_model()
        expectation_models: Dict[int, linear_model.LogisticRegression] = {}
        clustering = CustomAgglomerativeClustering(
            self.sentence_transformer,
            self.feature_generator,
        )

        def process_features(features, input_sentence,):
            processed_input_sentence = preprocess_sentence(input_sentence)
            processed_question = preprocess_sentence(features["question"])
            processed_ia = preprocess_sentence(features["ideal"])

            features_list = LRExpectationClassifier.calculate_features(
                processed_question,
                input_sentence,
                processed_input_sentence,
                processed_ia,
                self.sentence_transformer,
                self.feature_generator,
                [],
                [],
                clustering,
                ClassifierMode.TRAIN,
            )
            return features_list

        all_features = list(
            data.apply(
                lambda row: process_features(
                    json.loads(row["exp_data"]), row["text"]
                ),
                axis=1,
            )
        )
        train_y = np.array(LRExpectationClassifier.encode_y(data["label"]))
        model.fit(all_features, train_y)
        results_loocv = model_selection.cross_val_score(
            model, all_features, train_y, cv=LeaveOneOut(), scoring="accuracy"
        )
        accuracy = results_loocv.mean()
        expectation_models[data["exp_num"].iloc[0]] = model
        dao.save_default_pickle(
            DefaultModelSaveReq(
                arch=ARCH_LR_TRANS_EMB_CLASSIFIER,
                filename="models_by_expectation_num.pkl",
                model=expectation_models,
            )
        )
        return dao.create_default_training_result(
            ARCH_LR_TRANS_EMB_CLASSIFIER, ExpectationTrainingResult(accuracy=accuracy)
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
        clustering = CustomAgglomerativeClustering(
            self.sentence_transformer,
            self.feature_generator,
        )
        config_updated = train_input.config.clone()
        expectation_results: List[ExpectationTrainingResult] = []
        expectation_models: Dict[int, linear_model.LogisticRegression] = {}
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
            ideal_answer = LRExpectationClassifier.initialize_ideal_answer(
                processed_data
            )
            good = train_input.config.get_expectation_feature(exp_num, "good", [])
            bad = train_input.config.get_expectation_feature(exp_num, "bad", [])

            pattern: Dict[str, List[str]] = {"good": [], "bad": []}
            if self.train_quality > 0:
                data, candidates = clustering.generate_feature_candidates(
                    np.array(processed_data)[np.array(train_y) == "good"],
                    np.array(processed_data)[np.array(train_y) == "bad"],
                    self.train_quality,
                )
                pattern = clustering.select_feature_candidates(
                    data, candidates, train_x, train_y
                )

            config_updated.expectations[exp_num].features = {
                "good": good,
                "bad": bad,
                "patterns_good": pattern["good"],
                "patterns_bad": pattern["bad"],
                FEATURE_LENGTH_RATIO: feature_length_ratio_enabled(),
                FEATURE_REGEX_AGGREGATE_DISABLED: feature_regex_aggregate_disabled(),
            }

            features = [
                np.array(
                    LRExpectationClassifier.calculate_features(
                        processed_question,
                        raw_example,
                        example,
                        ideal_answer,
                        self.sentence_transformer,
                        self.feature_generator,
                        good,
                        bad,
                        clustering,
                        mode=ClassifierMode.TRAIN,
                        expectation_config=train_input.config.expectations[exp_num],
                        patterns=pattern["good"] + pattern["bad"],
                    )
                )
                for raw_example, example in zip(train_x, processed_data)
            ]
            train_y = np.array(LRExpectationClassifier.encode_y(train_y))
            model = LRExpectationClassifier.initialize_model()
            model.fit(features, train_y)
            results_loocv = model_selection.cross_val_score(
                model, features, train_y, cv=LeaveOneOut(), scoring="accuracy"
            )
            expectation_results.append(
                ExpectationTrainingResult(accuracy=results_loocv.mean())
            )
            expectation_models[exp_num] = model
        dao.save_pickle(
            ModelSaveReq(
                arch=ARCH_LR_TRANS_EMB_CLASSIFIER,
                lesson=train_input.lesson,
                filename="models_by_expectation_num.pkl",
                model=expectation_models,
            )
        )
        dao.save_config(
            QuestionConfigSaveReq(
                arch=ARCH_LR_TRANS_EMB_CLASSIFIER,
                lesson=train_input.lesson,
                config=config_updated,
            )
        )
        return TrainingResult(
            expectations=expectation_results,
            lesson=train_input.lesson,
            models=dao.get_model_root(
                ArchLesson(arch=ARCH_LR_TRANS_EMB_CLASSIFIER, lesson=train_input.lesson)
            ),
        )
