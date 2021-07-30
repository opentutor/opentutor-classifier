#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
from typing import List

import pytest
import responses

from opentutor_classifier import (
    ExpectationTrainingResult,
    ARCH_LR_CLASSIFIER,
    ARCH_LR_TRANS_EMB_CLASSIFIER,
)
from opentutor_classifier.config import confidence_threshold_default
from .utils import (
    assert_testset_accuracy,
    assert_train_expectation_results,
    create_and_test_classifier,
    fixture_path,
    read_example_testset,
    test_env_isolated,
    train_classifier,
    train_default_classifier,
    _TestExpectation,
)

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.mark.parametrize("lesson", [("question1"), ("question2")])
def test_outputs_models_at_specified_root(
    tmpdir, data_root: str, shared_root: str, lesson: str
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, lesson=lesson
    ) as test_config:
        result = train_classifier(lesson, test_config)
        assert path.exists(path.join(result.models, "models_by_expectation_num.pkl"))
        assert path.exists(path.join(result.models, "config.yaml"))


@pytest.mark.parametrize(
    "arch,expected_model_file_name",
    [
        (ARCH_LR_CLASSIFIER, "models_by_expectation_num.pkl"),
        (ARCH_LR_TRANS_EMB_CLASSIFIER, "models_by_expectation_num.pkl"),
    ],
)
def test_outputs_models_at_specified_model_root_for_default_model(
    arch: str, expected_model_file_name: str, tmpdir, data_root: str, shared_root: str
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, lesson="default"
    ) as test_config:
        result = train_default_classifier(test_config)
        assert path.exists(path.join(result.models, expected_model_file_name))


def _test_train_and_predict(
    lesson: str,
    arch: str,
    # confidence_threshold for now determines whether an answer
    # is really classified as GOOD/BAD (confidence >= threshold)
    # or whether it is interpretted as NEUTRAL (confidence < threshold)
    confidence_threshold: float,
    expected_training_result: List[ExpectationTrainingResult],
    expected_accuracy: float,
    tmpdir,
    data_root: str,
    shared_root: str,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)
        assert_train_expectation_results(
            train_result.expectations, expected_training_result
        )
        testset = read_example_testset(
            lesson, confidence_threshold=confidence_threshold
        )
        assert_testset_accuracy(
            arch,
            train_result.models,
            shared_root,
            testset,
            expected_accuracy=expected_accuracy,
        )


@pytest.mark.only
@pytest.mark.parametrize(
    "example,arch,confidence_threshold,expected_training_result,expected_accuracy",
    [
        # (
        #     "ies-rectangle",
        #     ARCH_LR_CLASSIFIER,
        #     CONFIDENCE_THRESHOLD_DEFAULT,
        #     [
        #         ExpectationTrainingResult(accuracy=0.89),
        #         ExpectationTrainingResult(accuracy=0.92),
        #         ExpectationTrainingResult(accuracy=0.95),
        #     ],
        #     1,
        # ),
        (
            "candles",
            ARCH_LR_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [
                ExpectationTrainingResult(accuracy=0.70),
                ExpectationTrainingResult(accuracy=0.76),
                ExpectationTrainingResult(accuracy=0.82),
                ExpectationTrainingResult(accuracy=0.75),
            ],
            0.44,
        ),
        # (
        #     "ies_auction",
        #     ARCH_LR_CLASSIFIER,
        #     CONFIDENCE_THRESHOLD_DEFAULT,
        #     [
        #         ExpectationTrainingResult(accuracy=0.79),
        #         ExpectationTrainingResult(accuracy=0.78),
        #         ExpectationTrainingResult(accuracy=0.92),
        #         ExpectationTrainingResult(accuracy=0.92),
        #     ],
        #     0.73,
        # ),
        # (
        #     "ies_bus_and_seats",
        #     ARCH_LR_CLASSIFIER,
        #     CONFIDENCE_THRESHOLD_DEFAULT,
        #     [
        #         ExpectationTrainingResult(accuracy=0.90),
        #         ExpectationTrainingResult(accuracy=0.84),
        #         ExpectationTrainingResult(accuracy=0.91),
        #         ExpectationTrainingResult(accuracy=0.92),
        #     ],
        #     0.84,
        # ),
        # (
        #     "ies_house",
        #     ARCH_LR_CLASSIFIER,
        #     CONFIDENCE_THRESHOLD_DEFAULT,
        #     [
        #         ExpectationTrainingResult(accuracy=0.92),
        #         ExpectationTrainingResult(accuracy=0.84),
        #         ExpectationTrainingResult(accuracy=0.85),
        #         ExpectationTrainingResult(accuracy=0.80),
        #     ],
        #     0.83,
        # ),
        # (
        #     "ies_pitcher",
        #     ARCH_LR_CLASSIFIER,
        #     CONFIDENCE_THRESHOLD_DEFAULT,
        #     [
        #         ExpectationTrainingResult(accuracy=0.81),
        #         ExpectationTrainingResult(accuracy=0.95),
        #         ExpectationTrainingResult(accuracy=0.68),
        #     ],
        #     0.62,
        # ),
        # (
        #     "ies_robot_remedy",
        #     ARCH_LR_CLASSIFIER,
        #     CONFIDENCE_THRESHOLD_DEFAULT,
        #     [
        #         ExpectationTrainingResult(accuracy=0.93),
        #         ExpectationTrainingResult(accuracy=0.64),
        #         ExpectationTrainingResult(accuracy=0.97),
        #     ],
        #     0.82,
        # ),
        # (
        #     "navy_integrity",
        #     ARCH_LR_CLASSIFIER,
        #     CONFIDENCE_THRESHOLD_DEFAULT,
        #     [
        #         ExpectationTrainingResult(accuracy=0.93),
        #         ExpectationTrainingResult(accuracy=0.64),
        #         ExpectationTrainingResult(accuracy=0.93),
        #     ],
        #     0.90,
        # ),
        # (
        #     "suicide",
        #     ARCH_LR_CLASSIFIER,
        #     CONFIDENCE_THRESHOLD_DEFAULT,
        #     [
        #         ExpectationTrainingResult(accuracy=0.76),
        #         ExpectationTrainingResult(accuracy=0.70),
        #     ],
        #     0.53,
        # ),
    ],
)
@pytest.mark.slow
def test_train_and_predict_slow(
    example: str,
    arch: str,
    # confidence_threshold for now determines whether an answer
    # is really classified as GOOD/BAD (confidence >= threshold)
    # or whether it is interpretted as NEUTRAL (confidence < threshold)
    confidence_threshold: float,
    expected_training_result: List[ExpectationTrainingResult],
    expected_accuracy: float,
    tmpdir,
    data_root: str,
    shared_root: str,
):
    _test_train_and_predict(
        example,
        arch,
        confidence_threshold,
        expected_training_result,
        expected_accuracy,
        tmpdir,
        data_root,
        shared_root,
    )


def _test_train_and_predict_specific_answers_slow(
    lesson: str,
    arch: str,
    evaluate_input_list: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_result: List[_TestExpectation],
    tmpdir,
    data_root: str,
    shared_root: str,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)
        assert_train_expectation_results(
            train_result.expectations, expected_training_result
        )
        for evaluate_input, ans in zip(evaluate_input_list, expected_evaluate_result):
            create_and_test_classifier(
                lesson,
                path.split(path.abspath(train_result.models))[0],
                shared_root,
                evaluate_input,
                [ans],
                arch=arch,
            )


@pytest.mark.slow
@pytest.mark.parametrize(
    "lesson,arch,evaluate_input_list,expected_training_result,expected_evaluate_result",
    [
        (
            "suicide",
            ARCH_LR_TRANS_EMB_CLASSIFIER,
            [
                # "5",
                # "It is 3 and 7 and 4 and 0",
                # "30 and 74",
                # "they will still commit suicide",
                # "they have the same risk, you can't change it. it's genetic.",
                # "Long term they probably won't commit suicide",
                # "Their risk is the same",
                # "thirty seven by forty",
                # "forty by thirty seven",
                # "37 by forty",
                # "thirty-seven by forty",
                # "37.0 by 40.000",
                # "thirty seven by fourty",
            ],
            [ExpectationTrainingResult(accuracy=0.50)],
            [
                # _TestExpectation(evaluation="Bad", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Bad", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Bad", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                _TestExpectation(evaluation="Bad", score=0.99, expectation=0),
                # _TestExpectation(evaluation="Bad", score=0.50, expectation=0),
                # _TestExpectation(evaluation="Bad", score=0.50, expectation=1),
                # _TestExpectation(evaluation="Bad", score=0.50, expectation=1),
                # _TestExpectation(evaluation="Bad", score=0.50, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.50, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
            ],
        ),
    ],
)
def test_train_and_predict_specific_answers_slow(
    lesson: str,
    arch: str,
    evaluate_input_list: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_result: List[_TestExpectation],
    tmpdir,
    data_root: str,
    shared_root: str,
):
    _test_train_and_predict_specific_answers_slow(
        lesson,
        arch,
        evaluate_input_list,
        expected_training_result,
        expected_evaluate_result,
        tmpdir,
        data_root,
        shared_root,
    )


@responses.activate
@pytest.mark.parametrize(
    "arch",
    [
        ARCH_LR_CLASSIFIER,
    ],
)
def test_train_default(
    arch: str,
    data_root: str,
    shared_root: str,
    tmpdir,
):
    with test_env_isolated(
        tmpdir,
        data_root,
        shared_root,
        arch=arch,
        is_default_model=True,
        lesson="default",
    ) as config:
        train_default_classifier(config=config)
