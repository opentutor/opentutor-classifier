#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import environ, path
from typing import List

import pytest
import responses

from opentutor_classifier import (
    AnswerClassifierInput,
    ClassifierFactory,
    ClassifierConfig,
    ExpectationTrainingResult,
    TrainingConfig,
    TrainingOptions,
    ARCH_SVM_CLASSIFIER,
    ARCH_LR_CLASSIFIER,
)
from opentutor_classifier.config import confidence_threshold_default
from opentutor_classifier.training import (
    train_online,
    train_default_online,
)
from opentutor_classifier.utils import dict_to_config
from .utils import (
    assert_testset_accuracy,
    assert_train_expectation_results,
    create_and_test_classifier,
    fixture_path,
    read_example_testset,
    test_env_isolated,
    train_classifier,
    train_default_model,
    _TestExpectation,
)

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


@pytest.mark.parametrize("input_lesson", [("question1"), ("question2")])
def test_outputs_models_at_specified_root(
    tmpdir, data_root: str, shared_root: str, input_lesson: str
):
    with test_env_isolated(
        tmpdir, path.join(data_root, input_lesson), shared_root
    ) as test_config:
        result = train_classifier(test_config)
        assert path.exists(path.join(result.models, "models_by_expectation_num.pkl"))
        assert path.exists(path.join(result.models, "config.yaml"))


def test_outputs_models_at_specified_model_root_for_default_model(
    tmpdir, data_root: str, shared_root: str
):
    output_dir, _ = train_default_model(tmpdir, data_root, shared_root)
    assert path.exists(path.join(output_dir, "models_by_expectation_num.pkl"))


@pytest.mark.parametrize(
    "example,arch,confidence_threshold,expected_training_result,expected_accuracy",
    [
        (
            "question1",
            ARCH_SVM_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [
                ExpectationTrainingResult(accuracy=0.8),
                ExpectationTrainingResult(accuracy=0.7),
                ExpectationTrainingResult(accuracy=0.98),
            ],
            0.65,
        ),
        (
            "question2",
            ARCH_SVM_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [ExpectationTrainingResult(accuracy=0.98)],
            0.99,
        ),
        (
            "ies-rectangle",
            ARCH_LR_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [
                ExpectationTrainingResult(accuracy=0.90),
                ExpectationTrainingResult(accuracy=0.90),
                ExpectationTrainingResult(accuracy=0.90),
            ],
            0.85,
        ),
        (
            "ies-rectangle",
            ARCH_SVM_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [
                ExpectationTrainingResult(accuracy=0.92),
                ExpectationTrainingResult(accuracy=0.93),
                ExpectationTrainingResult(accuracy=0.93),
            ],
            0.8,
        ),
        (
            "candles",
            ARCH_LR_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [
                ExpectationTrainingResult(accuracy=0.82),
                ExpectationTrainingResult(accuracy=0.85),
                ExpectationTrainingResult(accuracy=0.82),
                ExpectationTrainingResult(accuracy=0.89),
            ],
            0.8,
        ),
        (
            "candles",
            ARCH_SVM_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [
                ExpectationTrainingResult(accuracy=0.82),
                ExpectationTrainingResult(accuracy=0.85),
                ExpectationTrainingResult(accuracy=0.82),
                ExpectationTrainingResult(accuracy=0.95),
            ],
            0.8,
        ),
    ],
)
def test_train_and_predict(
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
    with test_env_isolated(
        tmpdir, path.join(data_root, example), shared_root, arch=arch
    ) as test_config:
        train_result = train_classifier(test_config)
        assert path.exists(train_result.models)
        assert_train_expectation_results(
            train_result.expectations, expected_training_result
        )
        testset = read_example_testset(
            example, confidence_threshold=confidence_threshold
        )
        assert_testset_accuracy(
            arch,
            train_result.models,
            shared_root,
            testset,
            expected_accuracy=expected_accuracy,
        )


@responses.activate
@pytest.mark.parametrize(
    "lesson,arch,evaluate_inputs,expected_training_result,expected_evaluate_results",
    [],
)
def test_train_and_predict_multiple(
    lesson: str,
    arch: str,
    evaluate_inputs: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_results: List[List[_TestExpectation]],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    with test_env_isolated(
        tmpdir, path.join(data_root, lesson), shared_root, arch
    ) as test_config:
        train_result = train_classifier(test_config)
        assert path.exists(train_result.models)
        assert_train_expectation_results(
            train_result.expectations, expected_training_result
        )
        for evaluate_input, expected_evaluate_result in zip(
            evaluate_inputs, expected_evaluate_results
        ):
            create_and_test_classifier(
                train_result.models,
                shared_root,
                evaluate_input,
                expected_evaluate_result,
                arch=arch,
            )


@pytest.mark.parametrize(
    "lesson,arch,evaluate_input_list,expected_training_result,expected_evaluate_result",
    [
        (
            "question3",
            ARCH_SVM_CLASSIFIER,
            ["7 by 10", "38 by 39", "37x40", "12x23", "45 x 67"],
            [ExpectationTrainingResult(accuracy=0.98)],
            [
                _TestExpectation(evaluation="Bad", score=0.95, expectation=0),
                _TestExpectation(evaluation="Bad", score=0.95, expectation=0),
                _TestExpectation(evaluation="Good", score=0.92, expectation=0),
                _TestExpectation(evaluation="Bad", score=0.95, expectation=0),
                _TestExpectation(evaluation="Bad", score=0.95, expectation=0),
            ],
        ),
        # (
        #     "ies-rectangle",
        #     ARCH_LR_CLASSIFIER,
        #     [
        #         # "5",
        #         # "It is 3 and 7 and 4 and 0",
        #         # "30 and 74",
        #         "37 x 40",
        #         #"thirty seven by forty",
        #         "forty by thirty seven",
        #         # "37 by forty",
        #         # "thirty-seven by forty",
        #         # "37.0 by 40.000",
        #         # "thirty seven by fourty",
        #     ],
        #     [ExpectationTrainingResult(accuracy=0.90)],
        #     [
        #         # _TestExpectation(evaluation="Bad", score=0.80, expectation=2),
        #         # _TestExpectation(evaluation="Bad", score=0.80, expectation=2),
        #          _TestExpectation(evaluation="Bad", score=0.80, expectation=2),
        #         #_TestExpectation(evaluation="Good", score=0.80, expectation=2),
        #         _TestExpectation(evaluation="Good", score=0.80, expectation=2),
        #         # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
        #         # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
        #         # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
        #         # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
        #         # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
        #     ],
        # ),
    ],
)
def test_train_and_single_expectation_predict(
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
        tmpdir, path.join(data_root, lesson), shared_root, arch
    ) as test_config:
        train_result = train_classifier(test_config)
        assert path.exists(train_result.models)
        assert_train_expectation_results(
            train_result.expectations, expected_training_result
        )
        for evaluate_input, ans in zip(evaluate_input_list, expected_evaluate_result):
            create_and_test_classifier(
                train_result.models, shared_root, evaluate_input, [ans], arch=arch
            )


def _test_train_online(
    lesson: str,
    arch: str,
    evaluate_inputs: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_results: List[List[_TestExpectation]],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    lesson = environ.get("LESSON_OVERRIDE") or lesson
    with test_env_isolated(
        tmpdir, path.join(data_root, lesson), shared_root, arch=arch
    ) as test_config:
        train_result = train_online(
            lesson,
            TrainingConfig(shared_root=test_config.shared_root),
            TrainingOptions(
                archive_root=test_config.archive_root, output_dir=test_config.output_dir
            ),
            arch=arch,
        )
        assert_train_expectation_results(
            train_result.expectations, expected_training_result
        )
        assert path.exists(train_result.models)
        for evaluate_input, expected_evaluate_result in zip(
            evaluate_inputs, expected_evaluate_results
        ):
            create_and_test_classifier(
                train_result.models,
                shared_root,
                evaluate_input,
                expected_evaluate_result,
                arch=arch,
            )


@responses.activate
@pytest.mark.parametrize(
    "lesson,arch,evaluate_input,expected_training_result,expected_evaluate_result",
    [
        (
            "question1",
            ARCH_SVM_CLASSIFIER,
            "peer pressure can change your behavior",
            [
                ExpectationTrainingResult(accuracy=0.72),
                ExpectationTrainingResult(accuracy=0.18),
                ExpectationTrainingResult(accuracy=0.90),
            ],
            [
                _TestExpectation(evaluation="Good", score=0.98, expectation=0),
                _TestExpectation(evaluation="Bad", score=0.30, expectation=1),
                _TestExpectation(evaluation="Bad", score=0.30, expectation=2),
            ],
        ),
        (
            "question1",
            ARCH_LR_CLASSIFIER,
            "peer pressure can change your behavior",
            [
                ExpectationTrainingResult(accuracy=0.72),
                ExpectationTrainingResult(accuracy=0.18),
                ExpectationTrainingResult(accuracy=0.90),
            ],
            [
                _TestExpectation(evaluation="Good", score=0.71, expectation=0),
                _TestExpectation(evaluation="Bad", score=0.30, expectation=1),
                _TestExpectation(evaluation="Bad", score=0.30, expectation=2),
            ],
        ),
    ],
)
def test_train_online(
    lesson: str,
    arch: str,
    evaluate_input: str,
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_result: List[_TestExpectation],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    _test_train_online(
        lesson,
        arch,
        [evaluate_input],
        expected_training_result,
        [expected_evaluate_result],
        data_root,
        shared_root,
        tmpdir,
    )


@responses.activate
@pytest.mark.parametrize(
    "lesson,arch,evaluate_inputs,expected_training_result,expected_evaluate_results",
    [],
)
def test_multiple_train_online(
    lesson: str,
    arch: str,
    evaluate_inputs: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_results: List[List[_TestExpectation]],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    _test_train_online(
        lesson,
        arch,
        evaluate_inputs,
        expected_training_result,
        expected_evaluate_results,
        data_root,
        shared_root,
        tmpdir,
    )


@responses.activate
@pytest.mark.parametrize(
    "arch",
    [
        ARCH_SVM_CLASSIFIER,
        ARCH_LR_CLASSIFIER,
    ],
)
def test_train_default_online(
    arch: str,
    data_root: str,
    shared_root: str,
    tmpdir,
):
    with test_env_isolated(
        tmpdir,
        path.join(data_root, "default"),
        shared_root,
        arch=arch,
        is_default_model=True,
    ) as config:
        train_default_online(
            TrainingConfig(shared_root=config.shared_root),
            TrainingOptions(
                archive_root=config.archive_root, output_dir=config.output_dir
            ),
            arch,
        )


def test_trained_default_model_usable_for_inference(
    tmpdir, data_root: str, shared_root: str
):
    output_dir, result = train_default_model(tmpdir, data_root, shared_root)
    assert path.exists(output_dir)
    assert result.expectations[0].accuracy >= 0.72
    config_data = {
        "question": "What are the challenges to demonstrating integrity in a group?",
        "expectations": [
            {"ideal": "Peer pressure can cause you to allow inappropriate behavior"}
        ],
    }
    model_root, model_name = path.split(output_dir)
    classifier = ClassifierFactory().new_classifier(
        ClassifierConfig(
            model_name=model_name, model_roots=[model_root], shared_root=shared_root
        ),
        arch=ARCH_SVM_CLASSIFIER,
    )
    eval_result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence="peer pressure can change your behavior",
            config_data=dict_to_config(config_data),
            expectation=0,
        )
    )
    assert len(eval_result.expectation_results) == 1
    assert eval_result.expectation_results[0].evaluation == "Bad"
    assert round(eval_result.expectation_results[0].score, 2) == 0.0
