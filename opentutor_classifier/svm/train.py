#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict
from datetime import datetime
import json
from os import makedirs, path
import pickle
import shutil
import tempfile
from typing import Dict, List

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

from sklearn import model_selection, svm
from sklearn.model_selection import LeaveOneOut

from opentutor_classifier import (
    load_data,
    load_yaml,
    ExpectationFeatures,
    ExpectationTrainingResult,
    QuestionConfig,
    TrainingInput,
    TrainingResult,
)
from opentutor_classifier.api import fetch_training_data, GRAPHQL_ENDPOINT
from opentutor_classifier.log import logger
from opentutor_classifier.stopwords import STOPWORDS
from .predict import (  # noqa: F401
    preprocess_sentence,
    SVMAnswerClassifier,
    SVMExpectationClassifier,
)
from .word2vec import find_or_load_word2vec


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


def _save(model_instances, filename):
    logger.info(f"saving models to {filename}")
    pickle.dump(model_instances, open(filename, "wb"))


class SVMAnswerClassifierTraining:
    def __init__(self, shared_root: str = "shared"):
        self.word2vec = find_or_load_word2vec(path.join(shared_root, "word2vec.bin"))
        self.model_obj = SVMExpectationClassifier()
        self.accuracy: Dict[int, int] = {}

    def default_train_all(
        self, data_root: str = "data", output_dir: str = "output"
    ) -> Dict:
        try:
            training_data = load_data(path.join(data_root, "default", "training.csv"))
        except Exception:
            training_data = self.model_obj.combine_dataset(data_root)
        model = self.model_obj.initialize_model()
        index2word_set = set(self.word2vec.index2word)
        output_dir = path.abspath(output_dir)
        makedirs(output_dir, exist_ok=True)
        expectation_models: Dict[int, svm.SVC] = {}

        def process_features(features, input_sentence, index2word_set):
            processed_input_sentence = preprocess_sentence(input_sentence)
            processed_question = preprocess_sentence(features["question"])
            processed_ia = preprocess_sentence(features["ideal"])

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
        return accuracy

    def train(
        self,
        train_input: TrainingInput,
        add_ideal_answers_to_training_data=True,
        archive_root: str = "archive",
        output_dir: str = "output",
    ) -> TrainingResult:
        pd.set_option("display.max_colwidth", 1000)
        question = str(train_input.config.get("question") or "")
        config_expectations = train_input.config.get("expectations") or []
        if not question:
            raise ValueError("config must have a 'question'")
        train_data = (
            pd.DataFrame(
                [
                    [i, x.get("ideal"), "good"]
                    for i, x in enumerate(config_expectations)
                    if x.get("ideal")
                ],
                columns=["exp_num", "text", "label"],
            ).append(train_input.data, ignore_index=True)
            if add_ideal_answers_to_training_data
            else train_input.data
        )
        split_training_sets: dict = defaultdict(int)
        for i, exp_num in enumerate(train_data["exp_num"]):
            if exp_num not in split_training_sets:
                split_training_sets[exp_num] = [[], []]
            split_training_sets[exp_num][0].append(train_data["text"][i])
            split_training_sets[exp_num][1].append(train_data["label"][i])
        index2word_set: set = set(self.word2vec.index2word)
        expectation_features_objects = []
        expectation_results: List[ExpectationTrainingResult] = []
        expectation_models: Dict[int, svm.SVC] = {}
        for exp_num, (train_x, train_y) in split_training_sets.items():
            processed_data = _preprocess_trainx(train_x)
            processed_question = preprocess_sentence(question)
            ia = self.model_obj.initialize_ideal_answer(processed_data)
            good_regex = self.model_obj.get_regex(
                exp_num, config_expectations, "good_regex"
            )
            bad_regex = self.model_obj.get_regex(
                exp_num, config_expectations, "bad_regex"
            )
            expectation_features_objects.append(
                ExpectationFeatures(
                    ideal=ia, good_regex=good_regex, bad_regex=bad_regex
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
            expectation_results.append(
                ExpectationTrainingResult(accuracy=results_loocv.mean())
            )
            expectation_models[exp_num] = model
        tmp_save_dir = tempfile.mkdtemp()
        _save(
            expectation_models, path.join(tmp_save_dir, "models_by_expectation_num.pkl")
        )
        QuestionConfig(
            question=question, expectations=expectation_features_objects
        ).write_to(path.join(tmp_save_dir, "config.yaml"))
        output_dir = path.abspath(output_dir)
        archive_path = _archive_if_exists(output_dir, archive_root)
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


def train_data_root(
    archive_root: str = "archive",
    data_root="data",
    output_dir: str = "out",
    shared_root="shared",
):
    return SVMAnswerClassifierTraining(shared_root=shared_root).train(
        TrainingInput(
            config=load_yaml(path.join(data_root, "config.yaml")),
            data=load_data(path.join(data_root, "training.csv")),
        ),
        archive_root=archive_root,
        output_dir=output_dir,
    )


def train_online(
    lesson: str,
    archive_root: str = "archive",
    fetch_training_data_url=GRAPHQL_ENDPOINT,
    output_dir: str = "out",
    shared_root="shared",
) -> TrainingResult:
    return SVMAnswerClassifierTraining(shared_root=shared_root).train(
        fetch_training_data(lesson), archive_root=archive_root, output_dir=output_dir
    )


def train_default_classifier(
    data_root="data", output_dir: str = "out", shared_root="shared"
):
    svm_answer_classifier_training = SVMAnswerClassifierTraining(
        shared_root=shared_root
    )
    accuracy = svm_answer_classifier_training.default_train_all(
        data_root=data_root, output_dir=output_dir
    )
    return accuracy
