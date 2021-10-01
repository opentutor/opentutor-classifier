#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict
import json
from opentutor_classifier.utils import prop_bool
from os import path

from typing import Dict, List
from .constants import (
    ARCHETYPE_BAD,
    ARCHETYPE_GOOD,
    BAD,
    FEATURE_REGEX_AGGREGATE_DISABLED,
    FEATURE_ARCHETYPE_ENABLED,
    FEATURE_PATTERNS_ENABLED,
    GOOD,
    PATTERNS_BAD,
    PATTERNS_GOOD,
)
from .features import feature_regex_aggregate_disabled
from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model
from sklearn.model_selection import LeaveOneOut

from opentutor_classifier import DataDao, QuestionConfig
from opentutor_classifier import (
    ARCH_LR2_CLASSIFIER,
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
from opentutor_classifier.config import (
    LABEL_BAD,
    LABEL_GOOD,
    get_train_quality_default,
    PROP_TRAIN_QUALITY,
)
from opentutor_classifier.log import logger

from .constants import FEATURE_LENGTH_RATIO
from .expectations import LRExpectationClassifier
from .features import feature_length_ratio_enabled, preprocess_sentence

from opentutor_classifier.word2vec import find_or_load_word2vec
from opentutor_classifier.spacy_preprocessor import SpacyPreprocessor

from .clustering_features import CustomDBScanClustering


def _preprocess_trainx(
    data: List[str], preprocessor: SpacyPreprocessor
) -> List[List[str]]:
    pre_processed_dataset = [preprocess_sentence(entry, preprocessor) for entry in data]
    return pre_processed_dataset


class LRAnswerClassifierTraining(AnswerClassifierTraining):
    def __init__(self):
        self.accuracy: Dict[str, int] = {}
        self.word2vec: Word2VecKeyedVectors = None
        self.shared_root = None

    def configure(self, config: TrainingConfig) -> AnswerClassifierTraining:
        self.word2vec = find_or_load_word2vec(
            path.join(config.shared_root, "word2vec.bin")
        )
        self.train_quality = config.properties.get(
            PROP_TRAIN_QUALITY, get_train_quality_default()
        )
        self.shared_root = config.shared_root
        return self

    def train_default(self, data: pd.DataFrame, dao: DataDao) -> TrainingResult:
        model = LRExpectationClassifier.initialize_model()
        index2word_set = set(self.word2vec.index_to_key)
        expectation_models: Dict[int, linear_model.LogisticRegression] = {}
        clustering = CustomDBScanClustering(self.word2vec, index2word_set)
        preprocessor = SpacyPreprocessor(self.shared_root)

        def process_features(features, input_sentence, index2word_set, preprocessor):
            processed_input_sentence = preprocess_sentence(input_sentence, preprocessor)
            processed_question = preprocess_sentence(features["question"], preprocessor)
            processed_ia = preprocess_sentence(features["ideal"], preprocessor)

            features_list = LRExpectationClassifier.calculate_features(
                processed_question,
                input_sentence,
                processed_input_sentence,
                processed_ia,
                self.word2vec,
                index2word_set,
                [],
                [],
                clustering,
                ClassifierMode.TRAIN,
                preprocessor,
                feature_archetype_enabled=False,
                feature_patterns_enabled=False,
            )
            return features_list

        all_features = list(
            data.apply(
                lambda row: process_features(
                    json.loads(row["exp_data"]),
                    row["text"],
                    index2word_set,
                    preprocessor,
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
                arch=ARCH_LR2_CLASSIFIER,
                filename="models_by_expectation_num.pkl",
                model=expectation_models,
            )
        )
        dao.save_default_config(arch=ARCH_LR2_CLASSIFIER, config=QuestionConfig())
        return dao.create_default_training_result(
            ARCH_LR2_CLASSIFIER,
            ExpectationTrainingResult(expectation_id="", accuracy=accuracy),
        )

    def train(self, train_input: TrainingInput, dao: DataDao) -> TrainingResult:
        question = train_input.config.question or ""
        if not question:
            raise ValueError("config must have a 'question'")
        train_data = (
            pd.DataFrame(
                [
                    [x.expectation_id, x.ideal, LABEL_GOOD]
                    for i, x in enumerate(train_input.config.expectations)
                    if x.ideal
                ],
                columns=["exp_num", "text", "label"],
            ).append(train_input.data, ignore_index=True)
        ).sort_values(by=["exp_num"], ignore_index=True)
        split_training_sets: dict = defaultdict(int)
        for i, exp_num in enumerate(train_data["exp_num"]):
            label = str(train_data["label"][i]).lower().strip()
            if label not in [LABEL_GOOD, LABEL_BAD]:
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
        clustering = CustomDBScanClustering(self.word2vec, index2word_set)
        config_updated = train_input.config.clone()
        expectation_results: List[ExpectationTrainingResult] = []
        expectation_models: Dict[str, linear_model.LogisticRegression] = {}
        supergoodanswer = ""
        for exp_num in split_training_sets.keys():
            ideal = train_input.config.get_expectation_ideal(exp_num)
            if ideal:
                supergoodanswer = supergoodanswer + ideal
            else:
                supergoodanswer = supergoodanswer + split_training_sets[exp_num][0][0]

        preprocessor = SpacyPreprocessor(self.shared_root)
        for exp_num, (train_x, train_y) in split_training_sets.items():
            train_x.append(supergoodanswer)
            train_y.append(GOOD)
            processed_data = _preprocess_trainx(train_x, preprocessor)
            processed_question = preprocess_sentence(question, preprocessor)
            ideal_answer = LRExpectationClassifier.initialize_ideal_answer(
                processed_data
            )
            good = train_input.config.get_expectation_feature(exp_num, GOOD, [])
            bad = train_input.config.get_expectation_feature(exp_num, BAD, [])

            config_features = {
                GOOD: good,
                BAD: bad,
                FEATURE_LENGTH_RATIO: feature_length_ratio_enabled(),
                FEATURE_REGEX_AGGREGATE_DISABLED: feature_regex_aggregate_disabled(),
                FEATURE_ARCHETYPE_ENABLED: self.train_quality >= 1,
                FEATURE_PATTERNS_ENABLED: self.train_quality > 1,
            }

            if self.train_quality == 1:
                cluster_archetype = clustering.generate_feature_candidates(
                    np.array(processed_data)[np.array(train_y) == GOOD],
                    np.array(processed_data)[np.array(train_y) == BAD],
                    self.train_quality,
                )
                config_features[ARCHETYPE_GOOD] = cluster_archetype[GOOD]
                config_features[ARCHETYPE_BAD] = cluster_archetype[BAD]
            elif self.train_quality > 1:
                (
                    data,
                    candidates,
                    cluster_archetype,
                ) = clustering.generate_feature_candidates(
                    np.array(processed_data)[np.array(train_y) == GOOD],
                    np.array(processed_data)[np.array(train_y) == BAD],
                    self.train_quality,
                )
                pattern = clustering.select_feature_candidates(
                    data, candidates, train_x, train_y, preprocessor=preprocessor
                )

                config_features[ARCHETYPE_GOOD] = cluster_archetype[GOOD]
                config_features[ARCHETYPE_BAD] = cluster_archetype[BAD]
                config_features[PATTERNS_GOOD] = pattern[GOOD]
                config_features[PATTERNS_BAD] = pattern[BAD]

            config_updated.get_expectation(exp_num).features = config_features

            features = [
                np.array(
                    LRExpectationClassifier.calculate_features(
                        processed_question,
                        raw_example,
                        example,
                        ideal_answer,
                        self.word2vec,
                        index2word_set,
                        good,
                        bad,
                        clustering,
                        mode=ClassifierMode.TRAIN,
                        preprocessor=preprocessor,
                        feature_archetype_enabled=prop_bool(
                            FEATURE_ARCHETYPE_ENABLED, config_features
                        ),
                        feature_patterns_enabled=prop_bool(
                            FEATURE_PATTERNS_ENABLED, config_features
                        ),
                        expectation_config=train_input.config.get_expectation(exp_num),
                        patterns=config_features.get(PATTERNS_GOOD, [])
                        + config_features.get(PATTERNS_BAD, []),
                        archetypes=config_features.get(ARCHETYPE_GOOD, [])
                        + config_features.get(ARCHETYPE_BAD, []),
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
                ExpectationTrainingResult(
                    expectation_id=exp_num, accuracy=results_loocv.mean()
                )
            )
            expectation_models[exp_num] = model
        dao.save_pickle(
            ModelSaveReq(
                arch=ARCH_LR2_CLASSIFIER,
                lesson=train_input.lesson,
                filename="models_by_expectation_num.pkl",
                model=expectation_models,
            )
        )
        dao.save_config(
            QuestionConfigSaveReq(
                arch=ARCH_LR2_CLASSIFIER,
                lesson=train_input.lesson,
                config=config_updated,
            )
        )
        return TrainingResult(
            expectations=expectation_results,
            lesson=train_input.lesson,
            models=dao.get_model_root(
                ArchLesson(arch=ARCH_LR2_CLASSIFIER, lesson=train_input.lesson)
            ),
        )
