#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path

import pytest
import pandas as pd
import responses
import asyncio

from opentutor_classifier.config import confidence_threshold_default
from typing import List


from opentutor_classifier import (
    ClassifierFactory,
    ARCH_LR2_CLASSIFIER,
    TrainingConfig,
    TrainingInput,
    TrainingResult,
    ClassifierConfig,
    AnswerClassifierInput,
)
from opentutor_classifier.dao import (
    FileDataDao,
    load_data,
    load_config,
    _CONFIG_YAML,
    _TRAINING_CSV,
)

from .utils import (
    fixture_path,
    test_env_isolated,
    read_example_testset,
    run_classifier_testset,
)


CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()
_REP_FACTOR = [1, 2, 4, 8]


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


def predict_replicate_model(
    lesson: str,
    arch: str,
    dao: FileDataDao,
    shared_root: str,
    model_root,
    input,
    expectation,
) -> float:
    classifier = ClassifierFactory().new_classifier(
        ClassifierConfig(
            dao=dao,
            model_name=lesson,
            model_roots=[model_root],
            shared_root=shared_root,
        ),
        arch=arch,
    )
    class_input = AnswerClassifierInput(input, None, expectation)
    results = asyncio.run(classifier.evaluate(class_input))
    return results.expectation_results[0].score


def train_replicate_model(
    lesson: str,
    arch: str,
    data: pd.DataFrame,
    dao: FileDataDao,
    data_root: str,
    shared_root: str,
) -> TrainingResult:
    input = TrainingInput(
        lesson=lesson,
        config=load_config(path.join(data_root, lesson, _CONFIG_YAML)),
        data=data,  # dataframe
    )
    fac = ClassifierFactory()
    training = fac.new_training(
        arch=arch, config=TrainingConfig(shared_root=shared_root)
    )
    train_result = training.train(input, dao)
    return train_result


def find_testset_accuracy(arch, res, shared_root, testset):
    test_res = run_classifier_testset(arch, res.models, shared_root, testset)
    metrics = test_res.metrics()
    return metrics.accuracy


def assert_inc(
    accuracy_list: List[float],
) -> None:
    for i in range(1, len(accuracy_list)):
        assert accuracy_list[i] >= accuracy_list[i - 1]


@pytest.mark.slow
@responses.activate
@pytest.mark.parametrize(
    "lesson,arch,confidence_threshold,evaluate_input, evaluate_expectation",
    [
        (
            "mixture_toy",
            ARCH_LR2_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            "mixture a",
            "0",
        ),
    ],
)
def test_data_replication(
    tmpdir,
    data_root,
    shared_root,
    lesson: str,
    arch: str,
    confidence_threshold: float,
    evaluate_input: str,
    evaluate_expectation: str,
):
    with test_env_isolated(
        tmpdir,
        data_root,
        shared_root,
        arch=arch,
        lesson=lesson,
    ) as test_config:
        rep_factor = _REP_FACTOR
        data = load_data(path.join(data_root, lesson, _TRAINING_CSV))  # dao.py
        accuracy = []
        confidence = []
        testset = read_example_testset(
            lesson, confidence_threshold=confidence_threshold
        )
        for i in rep_factor:
            dao = FileDataDao(data_root, model_root=test_config.output_dir)
            data_list = [data] * i
            new_data = pd.concat(data_list)
            train_result = train_replicate_model(
                lesson, arch, new_data, dao, data_root, shared_root
            )
            acc = find_testset_accuracy(arch, train_result, shared_root, testset)
            accuracy.append(acc)
            predict_result = predict_replicate_model(
                lesson,
                arch,
                dao,
                shared_root,
                test_config.output_dir,
                evaluate_input,
                evaluate_expectation,
            )
            confidence.append(predict_result)
        assert_inc(accuracy)
        assert_inc(confidence)
