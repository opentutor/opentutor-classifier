#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict
from datetime import datetime
import json
import logging
from os import makedirs, path
import pickle
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model
from sklearn.model_selection import LeaveOneOut
from text_to_num import alpha2digit

from opentutor_classifier import (
    AnswerClassifierTraining,
    ExpectationConfig,
    ExpectationTrainingResult,
    QuestionConfig,
    TrainingConfig,
    TrainingInput,
    TrainingOptions,
    TrainingResult,
)
from opentutor_classifier.log import logger
from opentutor_classifier.stopwords import STOPWORDS
from .predict import (  # noqa: F401
    preprocess_sentence,
    LRAnswerClassifier,
    LRExpectationClassifier,
    preprocess_punctuations,
)

from opentutor_classifier.utils import load_data
from opentutor_classifier.word2vec import find_or_load_word2vec

from .clustering_features import generate_feature_candidates, select_feature_candidates


def _archive_if_exists(p: str, archive_root: str) -> str:
    if not path.exists(p):
        return ""
    makedirs(archive_root, exist_ok=True)
    archive_path = path.join(
        archive_root, f"{path.basename(p)}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    # can't use rename here because target is likely a network mount (e.g. S3 bucket)
    shutil.copytree(p, archive_path)
    shutil.rmtree(p)
    return archive_path


def _preprocess_trainx(data: List[str]) -> List[Tuple[Any, ...]]:
    pre_processed_dataset = []
    data = [entry.lower() for entry in data]
    data = [
        preprocess_punctuations(entry) for entry in data
    ]  # [ re.sub(r'[^\w\s]', '', entry) for entry in data ]
    data = [alpha2digit(entry, "en") for entry in data]
    data = [word_tokenize(entry) for entry in data]
    for entry in data:
        final_words = []
        for word, tag in pos_tag(entry):
            if word not in STOPWORDS:
                final_words.append(word)
        pre_processed_dataset.append(tuple(final_words))
    return np.array(pre_processed_dataset)


def _save(model_instances, filename):
    logger.info(f"saving models to {filename}")
    pickle.dump(model_instances, open(filename, "wb"))


class LRAnswerClassifierTraining(AnswerClassifierTraining):
    def __init__(self):
        self.model_obj = LRExpectationClassifier()
        self.accuracy: Dict[int, int] = {}

    def configure(self, config: TrainingConfig) -> AnswerClassifierTraining:
        self.word2vec = find_or_load_word2vec(
            path.join(config.shared_root, "word2vec.bin")
        )
        return self

    def _train_default(
        self,
        training_data: pd.DataFrame,
        config: TrainingConfig = None,
        opts: TrainingOptions = None,
    ) -> TrainingResult:
        model = self.model_obj.initialize_model()
        index2word_set = set(self.word2vec.index2word)
        output_dir = path.abspath((opts or TrainingOptions()).output_dir)
        makedirs(output_dir, exist_ok=True)
        expectation_models: Dict[int, linear_model.LogisticRegression] = {}

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
            )
            return features_list

        all_features = list(
            training_data.apply(
                lambda row: process_features(
                    json.loads(row["exp_data"]), row["text"], index2word_set
                ),
                axis=1,
            )
        )
        train_y = np.array(self.model_obj.encode_y(training_data["label"]))
        model.fit(all_features, train_y)
        results_loocv = model_selection.cross_val_score(
            model, all_features, train_y, cv=LeaveOneOut(), scoring="accuracy"
        )
        accuracy = results_loocv.mean()
        expectation_models[training_data["exp_num"].iloc[0]] = model
        _save(
            expectation_models, path.join(output_dir, "models_by_expectation_num.pkl")
        )
        # need to write config for default even though it's empty
        # or will get errors later on attempt to load
        QuestionConfig(question="").write_to(path.join(output_dir, "config.yaml"))
        return TrainingResult(
            lesson="default",
            expectations=[ExpectationTrainingResult(accuracy=accuracy)],
            models=output_dir,
            archive="",
        )

    def train_default(
        self,
        data_root: str = "data",
        config: TrainingConfig = None,
        opts: TrainingOptions = None,
    ) -> TrainingResult:
        try:
            training_data = load_data(path.join(data_root, "default", "training.csv"))
        except Exception:
            training_data = self.model_obj.combine_dataset(data_root)
        return self._train_default(
            training_data=training_data, config=config, opts=opts
        )

    def train_default_online(
        self,
        train_input: TrainingInput,
        config: TrainingConfig = None,
        opts: TrainingOptions = None,
    ) -> TrainingResult:
        return self._train_default(
            training_data=train_input.data, config=config, opts=opts
        )

    def train(
        self, train_input: TrainingInput, config: TrainingOptions
    ) -> TrainingResult:
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
            if config.add_ideal_answers_to_training_data
            else train_input.data
        ).sort_values(by=["exp_num"], ignore_index=True)
        split_training_sets: dict = defaultdict(int)
        for i, exp_num in enumerate(train_data["exp_num"]):
            label = str(train_data["label"][i]).lower().strip()
            if label not in ["good", "bad"]:
                logging.warning(
                    f"ignoring training-data row {i} with invalid label {label}"
                )
                continue
            if exp_num not in split_training_sets:
                split_training_sets[exp_num] = [[], []]
            split_training_sets[exp_num][0].append(
                str(train_data["text"][i]).lower().strip()
            )
            split_training_sets[exp_num][1].append(label)
        index2word_set: set = set(self.word2vec.index2word)
        conf_exps_out: List[ExpectationConfig] = []
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
            ideal_answer = self.model_obj.initialize_ideal_answer(processed_data)
            good = train_input.config.get_expectation_feature(exp_num, "good", [])
            bad = train_input.config.get_expectation_feature(exp_num, "bad", [])

            data, candidates = generate_feature_candidates(
                np.array(processed_data)[np.array(train_y) == "good"],
                np.array(processed_data)[np.array(train_y) == "bad"],
                self.word2vec,
                index2word_set,
            )

            pattern = select_feature_candidates(data, candidates)
            logging.warning(pattern)
            conf_exps_out.append(
                ExpectationConfig(
                    ideal=train_input.config.get_expectation_ideal(exp_num)
                    or " ".join(ideal_answer),
                    features=(dict(good=good, bad=bad, patterns=pattern)),
                )
            )

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
                        pattern,
                    )
                )
                for raw_example, example in zip(train_x, processed_data)
            ]
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
        tmp_save_dir = tempfile.mkdtemp()
        _save(
            expectation_models, path.join(tmp_save_dir, "models_by_expectation_num.pkl")
        )
        QuestionConfig(question=question, expectations=conf_exps_out).write_to(
            path.join(tmp_save_dir, "config.yaml")
        )
        output_dir = path.abspath(config.output_dir)
        archive_path = _archive_if_exists(output_dir, config.archive_root)
        makedirs(path.dirname(output_dir), exist_ok=True)
        logger.debug(f"copying results from {tmp_save_dir} to {output_dir}")
        # can't use rename here because target is likely a network mount (e.g. S3 bucket)
        shutil.copytree(tmp_save_dir, output_dir)
        shutil.rmtree(tmp_save_dir)
        return TrainingResult(
            archive=archive_path,
            expectations=expectation_results,
            lesson=train_input.lesson,
            models=output_dir,
        )
