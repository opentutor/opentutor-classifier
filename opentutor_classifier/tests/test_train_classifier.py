#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import environ, path, makedirs
import shutil
from typing import List

from freezegun import freeze_time
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
from opentutor_classifier.training import train_data_root, train_online
from opentutor_classifier.utils import dict_to_config, load_config
from .utils import (
    add_graphql_response,
    assert_testset_accuracy,
    assert_train_expectation_results,
    create_and_test_classifier,
    fixture_path,
    output_and_archive_for_test,
    read_example_testset,
    _TestExpectation,
    train_classifier,
    train_default_model,
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
    result = train_classifier(tmpdir, path.join(data_root, input_lesson), shared_root)
    assert path.exists(path.join(result.models, "models_by_expectation_num.pkl"))
    assert path.exists(path.join(result.models, "config.yaml"))


@pytest.mark.parametrize("input_lesson", [("question1")])
def test_training_archives_old_models(
    tmpdir, data_root: str, shared_root: str, input_lesson: str
):
    models_path: str = ""
    config_path: str = ""
    archive_root: str = ""
    data_path = path.join(data_root, input_lesson)
    config_1 = load_config(path.join(data_path, "config.yaml"))
    with freeze_time("20200901T012345"):
        result = train_classifier(tmpdir, data_path, shared_root)
        archive_root = path.join(path.dirname(result.models), "archive")
        models_path = path.join(result.models, "models_by_expectation_num.pkl")
        config_path = path.join(result.models, "config.yaml")
        assert path.exists(models_path)
        assert path.exists(config_path)
        assert load_config(config_path).question == config_1.question
    with freeze_time("20200902T123456"):
        # retrain the slightly altered data
        # to the the same output dir
        # it should copy the old model files to archive
        # and replace with newly trained
        test_data_path = path.join(tmpdir, "data_test", input_lesson)
        makedirs(path.dirname(test_data_path), exist_ok=True)
        shutil.copytree(data_path, test_data_path)
        # just change the config question, so we'll have something to test
        config_2_path = path.join(test_data_path, "config.yaml")
        config_2 = load_config(config_2_path)
        config_2.question = "question changed for train 1"
        config_2.write_to(config_2_path)
        result = train_data_root(
            data_root=test_data_path,
            config=TrainingConfig(shared_root=shared_root),
            opts=TrainingOptions(
                archive_root=archive_root,
                output_dir=result.models,
            ),
        )
        assert result.archive.endswith(f"{input_lesson}-20200902T123456")
        assert (
            load_config(path.join(result.archive, "config.yaml")).question
            == config_1.question
        )
        assert (
            load_config(path.join(result.models, "config.yaml")).question
            == config_2.question
        )


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
                ExpectationTrainingResult(accuracy=0.92),
                ExpectationTrainingResult(accuracy=0.93),
                ExpectationTrainingResult(accuracy=0.93),
            ],
            0.90,
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
                ExpectationTrainingResult(accuracy=0.95),
            ],
            0.9,
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
    train_result = train_classifier(
        tmpdir, path.join(data_root, example), shared_root, arch=arch
    )
    assert path.exists(train_result.models)
    assert_train_expectation_results(
        train_result.expectations, expected_training_result
    )
    testset = read_example_testset(example, confidence_threshold=confidence_threshold)
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
    train_result = train_classifier(
        tmpdir, path.join(data_root, lesson), shared_root, arch
    )
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
    train_result = train_classifier(
        tmpdir, path.join(data_root, lesson), shared_root, arch
    )
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
    add_graphql_response(lesson)
    output_dir, archive_root = output_and_archive_for_test(tmpdir, lesson)
    train_result = train_online(
        lesson,
        TrainingConfig(shared_root=shared_root),
        TrainingOptions(archive_root=archive_root, output_dir=output_dir),
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
                _TestExpectation(evaluation="Good", score=0.65, expectation=0),
                _TestExpectation(evaluation="Good", score=0.98, expectation=1),
                _TestExpectation(evaluation="Bad", score=0.46, expectation=2),
            ],
        ),
        (
            "example-2",
            ARCH_SVM_CLASSIFIER,
            "the hr team",
            [
                ExpectationTrainingResult(accuracy=0.87),
                ExpectationTrainingResult(accuracy=0.72),
            ],
            [
                _TestExpectation(evaluation="Good", score=0.98, expectation=0),
                _TestExpectation(evaluation="Bad", score=0.38, expectation=1),
            ],
        ),
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
                _TestExpectation(evaluation="Good", score=0.65, expectation=0),
                _TestExpectation(evaluation="Good", score=0.98, expectation=1),
                _TestExpectation(evaluation="Bad", score=0.46, expectation=2),
            ],
        ),
        (
            "ies-television",
            ARCH_SVM_CLASSIFIER,
            "percentages represent a ratio of parts per 100",
            [
                ExpectationTrainingResult(accuracy=0.67),
                ExpectationTrainingResult(accuracy=0.65),
                ExpectationTrainingResult(accuracy=0.89),
            ],
            [
                _TestExpectation(evaluation="Good", score=0.98, expectation=0),
                _TestExpectation(evaluation="Good", score=0.64, expectation=1),
                _TestExpectation(evaluation="Bad", score=0.0, expectation=2),
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
    "lesson,arch,evaluate_input,expected_training_result,expected_evaluate_result",
    [
        (
            "question1-with-unknown-props-in-config",
            ARCH_SVM_CLASSIFIER,
            "peer pressure can change your behavior",
            [
                ExpectationTrainingResult(accuracy=0.72),
                ExpectationTrainingResult(accuracy=0.18),
                ExpectationTrainingResult(accuracy=0.90),
            ],
            [
                _TestExpectation(evaluation="Good", score=0.66, expectation=0),
                _TestExpectation(evaluation="Good", score=0.98, expectation=1),
                _TestExpectation(evaluation="Bad", score=0.46, expectation=2),
            ],
        )
    ],
)
def test_train_online_works_if_config_has_unknown_props(
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
