#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict
import json

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model
from sklearn.model_selection import LeaveOneOut
from sentence_transformers import SentenceTransformer

from opentutor_classifier import DataDao
from opentutor_classifier import (
    ARCH_LR_TRANS_EMB_DIFF_CLASSIFIER,
    AnswerClassifierTraining,
    ArchLesson,
    DefaultModelSaveReq,
    ExpectationTrainingResult,
    ModelSaveReq,
    QuestionConfigSaveReq,
    TrainingConfig,
    TrainingInput,
    TrainingResult,
)
from opentutor_classifier.log import logger
from .expectations import LRExpectationClassifier

from .features import (
    preprocess_sentence,
)

from opentutor_classifier.sentence_transformer import find_or_load_sentence_transformer


def _preprocess_trainx(data: List[str]) -> List[List[str]]:
    pre_processed_dataset = [preprocess_sentence(entry) for entry in data]
    return np.array(pre_processed_dataset)


class LRAnswerClassifierTraining(AnswerClassifierTraining):
    def __init__(self):
        self.accuracy: Dict[int, int] = {}
        self.sentence_transformer: SentenceTransformer = None

    def configure(self, config: TrainingConfig) -> AnswerClassifierTraining:
        self.sentence_transformer = find_or_load_sentence_transformer(
            config.shared_root + "/../sentence-transformer"
        )
        return self

    def train_default(self, data: pd.DataFrame, dao: DataDao) -> TrainingResult:
        model = LRExpectationClassifier.initialize_model()
        expectation_models: Dict[int, linear_model.LogisticRegression] = {}

        def process_features(features, input_sentence):
            processed_ia = preprocess_sentence(features["ideal"])

            features_list = LRExpectationClassifier.calculate_features(
                input_sentence,
                processed_ia,
                self.sentence_transformer,
            )
            return features_list

        all_features = list(
            data.apply(
                lambda row: process_features(json.loads(row["exp_data"]), row["text"]),
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
                arch=ARCH_LR_TRANS_EMB_DIFF_CLASSIFIER,
                filename="models_by_expectation_num.pkl",
                model=expectation_models,
            )
        )
        return dao.create_default_training_result(
            ARCH_LR_TRANS_EMB_DIFF_CLASSIFIER,
            ExpectationTrainingResult(accuracy=accuracy),
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
            ideal_answer = LRExpectationClassifier.initialize_ideal_answer(
                processed_data
            )
            good = train_input.config.get_expectation_feature(exp_num, "good", [])
            bad = train_input.config.get_expectation_feature(exp_num, "bad", [])

            config_updated.expectations[exp_num].features = dict(
                good=good,
                bad=bad,
                patterns_good=[],  # pattern["good"],
                patterns_bad=[],  # pattern["bad"],
            )

            classifier = LRExpectationClassifier()
            classifier.set_ideal_emb(ideal_answer, self.sentence_transformer)

            features = [
                np.array(
                    classifier.calculate_features_train(
                        example,
                        self.sentence_transformer,
                    )
                )
                for _, example in zip(train_x, processed_data)
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
                arch=ARCH_LR_TRANS_EMB_DIFF_CLASSIFIER,
                lesson=train_input.lesson,
                filename="models_by_expectation_num.pkl",
                model=expectation_models,
            )
        )
        dao.save_config(
            QuestionConfigSaveReq(
                arch=ARCH_LR_TRANS_EMB_DIFF_CLASSIFIER,
                lesson=train_input.lesson,
                config=config_updated,
            )
        )
        return TrainingResult(
            expectations=expectation_results,
            lesson=train_input.lesson,
            models=dao.get_model_root(
                ArchLesson(
                    arch=ARCH_LR_TRANS_EMB_DIFF_CLASSIFIER, lesson=train_input.lesson
                )
            ),
        )
