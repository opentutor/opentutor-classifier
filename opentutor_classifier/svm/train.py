#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from collections import defaultdict
import numpy as np
from os import path, makedirs
from typing import Dict
from sklearn import model_selection, svm
from sklearn.model_selection import LeaveOneOut
import json


from opentutor_classifier import load_data, load_yaml, TrainingResult
from .dtos import InstanceConfig, InstanceExpectationFeatures
from .predict import SVMAnswerClassifier, SVMExpectationClassifier  # noqa: F401
from .word2vec import find_or_load_word2vec


class SVMAnswerClassifierTraining:
    def __init__(self, shared_root: str = "shared"):
        self.word2vec = find_or_load_word2vec(path.join(shared_root, "word2vec.bin"))
        self.model_obj = SVMExpectationClassifier()
        self.model_instances: Dict[int, svm.SVC] = {}
        self.accuracy: Dict[int, int] = {}

    def default_train_all(
        self, data_root: str = "data", output_dir: str = "output"
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
            processed_ia = self.model_obj.processing_single_sentence(features["ideal"])

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

    def train(self, lesson: str, output_dir: str = "output") -> TrainingResult:
        output_dir = path.abspath(output_dir)
        makedirs(output_dir, exist_ok=True)
        return TrainingResult(lesson=lesson, expectations=[])

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
            self.accuracy[exp_num] = results_loocv.mean() * 100.0
            self.model_instances[exp_num] = model
        self.model_obj.save(
            self.model_instances, path.join(output_dir, "models_by_expectation_num.pkl")
        )
        InstanceConfig(
            question=question, expectation_features=expectation_features_objects
        ).write_to(path.join(output_dir, "config.yaml"))
        return self.accuracy


def train_classifier(data_root="data", shared_root="shared", output_dir: str = "out"):
    svm_answer_classifier_training = SVMAnswerClassifierTraining(
        shared_root=shared_root
    )
    return svm_answer_classifier_training.train_all(
        data_root=data_root, output_dir=output_dir
    )


def train_classifier_online(
    lesson: str, shared_root="shared", output_dir: str = "out"
) -> TrainingResult:
    training = SVMAnswerClassifierTraining(
        shared_root=shared_root
    )
    result = training.train(lesson=lesson, output_dir=output_dir)
    # accuracy = svm_answer_classifier_training.train_all(
    #     data_root=data_root, output_dir=output_dir
    # )
    # return accuracy
    # return TrainingResult(lesson=lesson, expectations=[])
    return result


def train_default_classifier(
    data_root="data", shared_root="shared", output_dir: str = "out"
):
    svm_answer_classifier_training = SVMAnswerClassifierTraining(
        shared_root=shared_root
    )
    accuracy = svm_answer_classifier_training.default_train_all(
        data_root=data_root, output_dir=output_dir
    )
    return accuracy
