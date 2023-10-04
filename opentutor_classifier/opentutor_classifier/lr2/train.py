#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict
import json
import os
from opentutor_classifier.constants import DEPLOYMENT_MODE_OFFLINE

from opentutor_classifier.utils import prop_bool
from os import path, environ

from typing import Dict, List, Any

from opentutor_classifier.word2vec_wrapper import Word2VecWrapper, get_word2vec
from opentutor_classifier.dao import MODEL_ROOT_DEFAULT, _CONFIG_YAML
from .constants import (
    ARCHETYPE_BAD,
    ARCHETYPE_GOOD,
    BAD,
    FEATURE_REGEX_AGGREGATE_DISABLED,
    FEATURE_ARCHETYPE_ENABLED,
    FEATURE_PATTERNS_ENABLED,
    GOOD,
    MODEL_FILE_NAME,
    SLIM_EMBEDDING_FILE_NAME,
    PATTERNS_BAD,
    PATTERNS_GOOD,
)
from .features import feature_regex_aggregate_disabled
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
    EmbeddingSaveReq,
    ExpectationTrainingResult,
    ModelSaveReq,
    QuestionConfigSaveReq,
    TrainingConfig,
    TrainingInput,
    TrainingResult,
    ClassifierMode,
    ExpectationConfig,
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

from .clustering_features import CustomDBScanClustering

DEPLOYMENT_MODE = environ.get("DEPLOYMENT_MODE") or DEPLOYMENT_MODE_OFFLINE
MINIMUM_NUMBER_OF_GRADED_ENTRIES = 10


def _preprocess_trainx(data: List[str]) -> List[List[str]]:
    pre_processed_dataset = [preprocess_sentence(entry) for entry in data]
    return pre_processed_dataset


class LRAnswerClassifierTraining(AnswerClassifierTraining):
    def __init__(self):
        self.accuracy: Dict[str, int] = {}

    def configure(self, config: TrainingConfig) -> AnswerClassifierTraining:
        self.word2vec_wrapper: Word2VecWrapper = get_word2vec(
            path.join(config.shared_root, "word2vec.bin"),
            path.join(config.shared_root, "word2vec_slim.bin"),
        )
        self.train_quality = config.properties.get(
            PROP_TRAIN_QUALITY, get_train_quality_default()
        )

        return self

    def update_slim_embeddings(
        self,
        ideal: List[str],
        question: List[str],
        archtypes_good: List[str],
        archetypes_bad: List[str],
        patterns_good: List[str],
        patterns_bad: List[str],
    ):
        embeddings: Dict[str, List[float]] = dict()
        words_set = set()
        for word in ideal + question:
            words_set.add(word)
        for archetype in archtypes_good + archetypes_bad:
            for word in archetype.lower().split():
                words_set.add(word)
        for pattern in patterns_bad + patterns_good:
            for word in pattern.split(" + "):
                words_set.add(word)

        word_vecs = self.word2vec_wrapper.get_feature_vectors(words_set, True)

        for word in word_vecs.keys():
            embeddings[word] = list(map(lambda x: round(float(x), 9), word_vecs[word]))
        return embeddings

    def preload_feature_vectors_train_default(self, data: pd.DataFrame, index2word_set):
        """
        ONLINE USE ONLY
        This function preprocesses data and preloads their vectors for later use by process_features (via calculate_features)
        """
        if DEPLOYMENT_MODE == DEPLOYMENT_MODE_OFFLINE:
            return
        all_words = []
        for i, row in data.iterrows():
            input_sentence = row["text"]
            features = json.loads(row["exp_data"])
            processed_input_sentence = preprocess_sentence(input_sentence)
            processed_question = preprocess_sentence(features["question"])
            processed_ia = preprocess_sentence(features["ideal"])
            i_processed_input_sentence = set(processed_input_sentence).intersection(
                index2word_set
            )
            i_processed_question = set(processed_question).intersection(index2word_set)
            i_processed_ia = set(processed_ia).intersection(index2word_set)
            all_words.extend(
                [*i_processed_input_sentence, *i_processed_question, *i_processed_ia]
            )
        self.word2vec_wrapper.get_feature_vectors(set(all_words))

    def process_features(self, features, input_sentence, index2word_set, clustering):
        processed_input_sentence = preprocess_sentence(input_sentence)
        processed_question = preprocess_sentence(features["question"])
        processed_ia = preprocess_sentence(features["ideal"])

        features_list = LRExpectationClassifier.calculate_features(
            processed_question,
            input_sentence,
            processed_input_sentence,
            processed_ia,
            self.word2vec_wrapper,
            index2word_set,
            [],
            [],
            clustering,
            ClassifierMode.TRAIN,
            feature_archetype_enabled=False,
            feature_patterns_enabled=False,
        )
        return features_list

    def train_default(self, data: pd.DataFrame, dao: DataDao) -> TrainingResult:
        model = LRExpectationClassifier.initialize_model()
        index2word_set = set(self.word2vec_wrapper.index_to_key(True))
        expectation_models: Dict[int, linear_model.LogisticRegression] = {}
        clustering = CustomDBScanClustering(self.word2vec_wrapper, index2word_set)

        self.preload_feature_vectors_train_default(data, index2word_set)

        all_features = list(
            data.apply(
                lambda row: self.process_features(
                    json.loads(row["exp_data"]), row["text"], index2word_set, clustering
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
                filename=MODEL_FILE_NAME,
                model=expectation_models,
            )
        )
        dao.save_default_config(arch=ARCH_LR2_CLASSIFIER, config=QuestionConfig())
        return dao.create_default_training_result(
            ARCH_LR2_CLASSIFIER,
            ExpectationTrainingResult(expectation_id="", accuracy=accuracy),
        )

    def preload_all_feature_vectors_train(self, train_input: TrainingInput):
        """
        ONLINE USE ONLY
        preprocesses data and fetches their vectors from w2v in one batch to store in memory
        """
        if DEPLOYMENT_MODE == DEPLOYMENT_MODE_OFFLINE:
            return
        all_data_text = [
            *[
                x.ideal.lower().strip()
                for i, x in enumerate(train_input.config.expectations)
            ],
            *[x.lower().strip() for x in train_input.data["text"]],
        ]
        all_data_text_set = set(all_data_text)
        self.word2vec_wrapper.get_feature_vectors(all_data_text_set)

    def get_trainable_expectations(
        self, train_input: TrainingInput
    ) -> List[ExpectationConfig]:
        value_counts = train_input.data["exp_num"].value_counts()
        result: List[ExpectationConfig] = []

        for expectation in train_input.config.expectations:
            if (
                expectation.expectation_id in value_counts
                and value_counts[expectation.expectation_id]
                >= MINIMUM_NUMBER_OF_GRADED_ENTRIES
            ):
                result.append(expectation)

        return result

    def train(
        self, train_input: TrainingInput, dao: DataDao, developer_mode: bool = False
    ) -> TrainingResult:
        self.preload_all_feature_vectors_train(train_input)
        if not developer_mode:
            trainable_expectations = self.get_trainable_expectations(train_input)
        else:
            trainable_expectations = train_input.config.expectations
        question = train_input.config.question or ""
        if not question:
            raise ValueError("config must have a 'question'")
        train_data = (
            pd.DataFrame(
                [
                    [x.expectation_id, x.ideal, LABEL_GOOD]
                    for i, x in enumerate(trainable_expectations)
                    if x.ideal
                ],
                columns=["exp_num", "text", "label"],
            ).append(train_input.data, ignore_index=True)
        ).sort_values(by=["exp_num"], ignore_index=True)
        split_training_sets: dict = defaultdict(int)
        for i, exp_num in enumerate(train_data["exp_num"]):
            # skip non-trainable expectations
            if exp_num not in [e.expectation_id for e in trainable_expectations]:
                continue
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
        index2word_set: set = set(self.word2vec_wrapper.index_to_key(False))
        clustering = CustomDBScanClustering(self.word2vec_wrapper, index2word_set)
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

        slim_embeddings: Dict[str, List[float]] = dict()
        for exp_num, (train_x, train_y) in split_training_sets.items():
            train_x.append(supergoodanswer)
            train_y.append(GOOD)
            processed_data = _preprocess_trainx(train_x)
            processed_question = preprocess_sentence(question)
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
                    np.array(processed_data, dtype=object)[
                        np.array(train_y, dtype=object) == GOOD
                    ],
                    np.array(processed_data, dtype=object)[
                        np.array(train_y, dtype=object) == BAD
                    ],
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
                    np.array(processed_data, dtype=object)[
                        np.array(train_y, dtype=object) == GOOD
                    ],
                    np.array(processed_data, dtype=object)[
                        np.array(train_y, dtype=object) == BAD
                    ],
                    self.train_quality,
                )
                pattern = clustering.select_feature_candidates(
                    data, candidates, train_x, train_y
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
                        self.word2vec_wrapper,
                        index2word_set,
                        good,
                        bad,
                        clustering,
                        mode=ClassifierMode.TRAIN,
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
                    ),
                    dtype=object,
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
            slim_embeddings.update(
                self.update_slim_embeddings(
                    ideal_answer,
                    processed_question,
                    config_features.get(ARCHETYPE_GOOD, []),
                    config_features.get(ARCHETYPE_BAD, []),
                    config_features.get(PATTERNS_GOOD, []),
                    config_features.get(PATTERNS_BAD, []),
                )
            )
            expectation_models[exp_num] = model

        dao.save_pickle(
            ModelSaveReq(
                arch=ARCH_LR2_CLASSIFIER,
                lesson=train_input.lesson,
                filename=MODEL_FILE_NAME,
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
        dao.save_embeddings(
            EmbeddingSaveReq(
                arch=ARCH_LR2_CLASSIFIER,
                lesson=train_input.lesson,
                filename=SLIM_EMBEDDING_FILE_NAME,
                embedding=slim_embeddings,
            )
        )
        return TrainingResult(
            expectations=expectation_results,
            lesson=train_input.lesson,
            models=dao.get_model_root(
                ArchLesson(arch=ARCH_LR2_CLASSIFIER, lesson=train_input.lesson)
            ),
        )

    def upload_model(self, s3: Any, lesson: str, s3_bucket: str):
        s3.upload_file(
            os.path.join(
                MODEL_ROOT_DEFAULT,
                ARCH_LR2_CLASSIFIER,
                lesson,
                MODEL_FILE_NAME,
            ),
            s3_bucket,
            os.path.join(lesson, ARCH_LR2_CLASSIFIER, MODEL_FILE_NAME),
        )

        # upload model config
        s3.upload_file(
            os.path.join(MODEL_ROOT_DEFAULT, ARCH_LR2_CLASSIFIER, lesson, _CONFIG_YAML),
            s3_bucket,
            os.path.join(lesson, ARCH_LR2_CLASSIFIER, _CONFIG_YAML),
        )
        return
