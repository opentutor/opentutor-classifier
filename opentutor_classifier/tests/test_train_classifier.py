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
    ExpectationTrainingResult,
)
from opentutor_classifier.svm.predict import SVMAnswerClassifier
from opentutor_classifier.svm.train import (
    train_data_root,
    train_online,
)
from opentutor_classifier.svm.utils import load_config, dict_to_config
from .utils import (
    add_graphql_response,
    assert_train_expectation_results,
    create_and_test_classifier,
    fixture_path,
    output_and_archive_for_test,
    _TestExpectation,
    train_classifier,
    train_default_model,
)


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
            archive_root=archive_root,
            data_root=test_data_path,
            shared_root=shared_root,
            output_dir=result.models,
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
    "lesson,evaluate_input,expected_training_result,expected_evaluate_result",
    [
        (
            "question1",
            "peer pressure can change your behavior",
            [
                ExpectationTrainingResult(accuracy=0.8),
                ExpectationTrainingResult(accuracy=0.7),
                ExpectationTrainingResult(accuracy=0.98),
            ],
            [
                _TestExpectation(evaluation="Good", score=0.98, expectation=0),
                _TestExpectation(evaluation="Bad", score=0.68, expectation=1),
                _TestExpectation(evaluation="Bad", score=0.56, expectation=2),
            ],
        ),
        (
            "question2",
            "Current flows in the same direction as the arrow",
            [ExpectationTrainingResult(accuracy=0.98)],
            [_TestExpectation(evaluation="Good", score=0.95, expectation=0)],
        ),
    ],
)
def test_train_and_predict(
    lesson: str,
    evaluate_input: str,
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_result: List[_TestExpectation],
    tmpdir,
    data_root: str,
    shared_root: str,
):
    train_result = train_classifier(tmpdir, path.join(data_root, lesson), shared_root)
    assert path.exists(train_result.models)
    assert_train_expectation_results(
        train_result.expectations, expected_training_result
    )
    create_and_test_classifier(
        train_result.models, shared_root, evaluate_input, expected_evaluate_result
    )


@responses.activate
@pytest.mark.parametrize(
    "lesson,evaluate_inputs,expected_training_result,expected_evaluate_results",
    [
        (
            "ies-rectangle",
            [
                "The closer a ratio of the sides in a rectangle is to one, the more it looks like a square. The larger the sides of the rectangle, the less effect a 3 unit difference will have on the ratio of the sides. The correct answer is the rectangle with dimensions 37 ft by 40 ft.",
                "The closer a ratio of the sides in a rectangle is to one, the more it looks like a square.",
                "The larger the sides of the rectangle, the less effect a 3 unit difference will have on the ratio of the sides.",
                "The correct answer is the rectangle with dimensions 37 ft by 40 ft.",
            ],
            [
                ExpectationTrainingResult(accuracy=0.929),
                ExpectationTrainingResult(accuracy=0.956),
                ExpectationTrainingResult(accuracy=0.959),
            ],
            [
                [
                    _TestExpectation(evaluation="Good", score=0.09, expectation=0),
                    _TestExpectation(evaluation="Good", score=0.92, expectation=1),
                    _TestExpectation(evaluation="Good", score=0.9, expectation=2),
                ],
                [
                    _TestExpectation(evaluation="Good", score=0.14, expectation=0),
                    _TestExpectation(evaluation="Bad", score=0.939, expectation=1),
                    _TestExpectation(evaluation="Bad", score=0.976, expectation=2),
                ],
                [
                    _TestExpectation(evaluation="Bad", score=0.999, expectation=0),
                    _TestExpectation(evaluation="Good", score=0.907, expectation=1),
                    _TestExpectation(evaluation="Bad", score=0.976, expectation=2),
                ],
                [
                    _TestExpectation(evaluation="Bad", score=0.999, expectation=0),
                    _TestExpectation(evaluation="Bad", score=0.936, expectation=1),
                    _TestExpectation(evaluation="Good", score=0.899, expectation=2),
                ],
            ],
        ),
    ],
)
def test_train_and_predict_multiple(
    lesson: str,
    evaluate_inputs: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_results: List[List[_TestExpectation]],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    train_result = train_classifier(tmpdir, path.join(data_root, lesson), shared_root)
    assert path.exists(train_result.models)
    assert_train_expectation_results(
        train_result.expectations, expected_training_result
    )
    for evaluate_input, expected_evaluate_result in zip(
        evaluate_inputs, expected_evaluate_results
    ):
        create_and_test_classifier(
            train_result.models, shared_root, evaluate_input, expected_evaluate_result
        )


@pytest.mark.parametrize(
    "lesson,evaluate_input_list,expected_training_result,expected_evaluate_result",
    [
        (
            "question3",
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
    evaluate_input_list: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_result: List[_TestExpectation],
    tmpdir,
    data_root: str,
    shared_root: str,
):
    train_result = train_classifier(tmpdir, path.join(data_root, lesson), shared_root)
    assert path.exists(train_result.models)
    assert_train_expectation_results(
        train_result.expectations, expected_training_result
    )
    for evaluate_input, ans in zip(evaluate_input_list, expected_evaluate_result):
        create_and_test_classifier(
            train_result.models, shared_root, evaluate_input, [ans]
        )

@responses.activate
@pytest.mark.parametrize(
    "lesson,evaluate_inputs,expected_training_result,expected_evaluate_results",
    [
        (
            "ies-rectangle",
            [
                "The closer a ratio of the sides in a rectangle is to one, the more it looks like a square. The larger the sides of the rectangle, the less effect a 3 unit difference will have on the ratio of the sides. The correct answer is the rectangle with dimensions 37 ft by 40 ft.",
                "The closer a ratio of the sides in a rectangle is to one, the more it looks like a square.",
                "The larger the sides of the rectangle, the less effect a 3 unit difference will have on the ratio of the sides.",
                "The correct answer is the rectangle with dimensions 37 ft by 40 ft.",
            ],
            [
                ExpectationTrainingResult(accuracy=0.93),
                ExpectationTrainingResult(accuracy=0.96),
                ExpectationTrainingResult(accuracy=0.96),
            ],
            [
                [
                ExpectationClassifierResult(evaluation="Good", score=0.09, expectation=0),
                ExpectationClassifierResult(evaluation="Good", score=0.92, expectation=1),
                ExpectationClassifierResult(evaluation="Good", score=0.9, expectation=2),
                ],
                [
                ExpectationClassifierResult(evaluation="Good", score=0.14, expectation=0),
                ExpectationClassifierResult(evaluation="Bad", score=0.94, expectation=1),
                ExpectationClassifierResult(evaluation="Bad", score=0.98, expectation=2),
                ],
                [
                ExpectationClassifierResult(evaluation="Bad", score=1.0, expectation=0),
                ExpectationClassifierResult(evaluation="Good", score=0.91, expectation=1),
                ExpectationClassifierResult(evaluation="Bad", score=0.98, expectation=2),
                ],
                [
                ExpectationClassifierResult(evaluation="Bad", score=1.0, expectation=0),
                ExpectationClassifierResult(evaluation="Bad", score=0.94, expectation=1),
                ExpectationClassifierResult(evaluation="Good", score=0.9, expectation=2),
                ],
            ],
        ),
    ],
)
def test_train_and_predict_multiple(
    lesson: str,
    evaluate_inputs: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_results: List[List[ExpectationClassifierResult]],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    train_result = __train_classifier(tmpdir, path.join(data_root, lesson), shared_root)
    assert path.exists(train_result.models)
    assert_train_expectation_results(
        train_result.expectations, expected_training_result
    )
    for evaluate_input, expected_evaluate_result in zip(evaluate_inputs,expected_evaluate_results):
        create_and_test_classifier(
            train_result.models, shared_root, evaluate_input, expected_evaluate_result
        )

def _test_train_online(
    lesson: str,
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
        archive_root=archive_root,
        output_dir=output_dir,
        shared_root=shared_root,
    )
    assert_train_expectation_results(
        train_result.expectations, expected_training_result
    )
    assert path.exists(train_result.models)
    for evaluate_input, expected_evaluate_result in zip(
        evaluate_inputs, expected_evaluate_results
    ):
        create_and_test_classifier(
            train_result.models, shared_root, evaluate_input, expected_evaluate_result
        )


@responses.activate
@pytest.mark.parametrize(
    "lesson,evaluate_input,expected_training_result,expected_evaluate_result",
    [
        (
            "question1",
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
    evaluate_input: str,
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_result: List[_TestExpectation],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    _test_train_online(
        lesson,
        [evaluate_input],
        expected_training_result,
        [expected_evaluate_result],
        data_root,
        shared_root,
        tmpdir,
    )


@responses.activate
@pytest.mark.parametrize(
    "lesson,evaluate_inputs,expected_training_result,expected_evaluate_results",
    [
        (
            "ies-rectangle",
            [
                "The closer a ratio of the sides in a rectangle is to one, the more it looks like a square. The larger the sides of the rectangle, the less effect a 3 unit difference will have on the ratio of the sides. The correct answer is the rectangle with dimensions 37 ft by 40 ft.",
                "The closer a ratio of the sides in a rectangle is to one, the more it looks like a square.",
                "The larger the sides of the rectangle, the less effect a 3 unit difference will have on the ratio of the sides.",
                "The correct answer is the rectangle with dimensions 37 ft by 40 ft.",
            ],
            [
                ExpectationTrainingResult(accuracy=0.929),
                ExpectationTrainingResult(accuracy=0.956),
                ExpectationTrainingResult(accuracy=0.96),
            ],
            [
                [
                    _TestExpectation(evaluation="Good", score=0, expectation=0),
                    _TestExpectation(evaluation="Good", score=0.921, expectation=1),
                    _TestExpectation(evaluation="Good", score=0, expectation=2),
                ],
                [
                    _TestExpectation(evaluation="Good", score=0.14, expectation=0),
                    _TestExpectation(evaluation="Bad", score=0.919, expectation=1),
                    _TestExpectation(evaluation="Bad", score=0.987, expectation=2),
                ],
                [
                    _TestExpectation(evaluation="Bad", score=0.999, expectation=0),
                    _TestExpectation(evaluation="Good", score=0.919, expectation=1),
                    _TestExpectation(evaluation="Bad", score=0.988, expectation=2),
                ],
                [
                    _TestExpectation(evaluation="Bad", score=0.999, expectation=0),
                    _TestExpectation(evaluation="Bad", score=0.92, expectation=1),
                    _TestExpectation(evaluation="Good", score=0.79, expectation=2),
                ],
            ],
        ),
    ],
)
def test_multiple_train_online(
    lesson: str,
    evaluate_inputs: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_results: List[List[_TestExpectation]],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    _test_train_online(
        lesson,
        evaluate_inputs,
        expected_training_result,
        expected_evaluate_results,
        data_root,
        shared_root,
        tmpdir,
    )

@responses.activate
@pytest.mark.parametrize(
    "lesson,evaluate_inputs,expected_training_result,expected_evaluate_results",
    [
        (
            "ies-rectangle",
            [
                "The closer a ratio of the sides in a rectangle is to one, the more it looks like a square. The larger the sides of the rectangle, the less effect a 3 unit difference will have on the ratio of the sides. The correct answer is the rectangle with dimensions 37 ft by 40 ft.",
                "The closer a ratio of the sides in a rectangle is to one, the more it looks like a square.",
                "The larger the sides of the rectangle, the less effect a 3 unit difference will have on the ratio of the sides.",
                "The correct answer is the rectangle with dimensions 37 ft by 40 ft.",
            ],
            [
                ExpectationTrainingResult(accuracy=0.93),
                ExpectationTrainingResult(accuracy=0.96),
                ExpectationTrainingResult(accuracy=0.96),
            ],
            [
                [
                ExpectationClassifierResult(evaluation="Good", score=0.09, expectation=0),
                ExpectationClassifierResult(evaluation="Good", score=0.93, expectation=1),
                ExpectationClassifierResult(evaluation="Good", score=0.84, expectation=2),
                ],
                [
                ExpectationClassifierResult(evaluation="Good", score=0.14, expectation=0),
                ExpectationClassifierResult(evaluation="Bad", score=0.92, expectation=1),
                ExpectationClassifierResult(evaluation="Bad", score=0.99, expectation=2),
                ],
                [
                ExpectationClassifierResult(evaluation="Bad", score=1.0, expectation=0),
                ExpectationClassifierResult(evaluation="Good", score=0.92, expectation=1),
                ExpectationClassifierResult(evaluation="Bad", score=0.99, expectation=2),
                ],
                [
                ExpectationClassifierResult(evaluation="Bad", score=1.0, expectation=0),
                ExpectationClassifierResult(evaluation="Bad", score=0.92, expectation=1),
                ExpectationClassifierResult(evaluation="Good", score=0.79, expectation=2),
                ],
            ],
        ),
    ],
)
def test_multiple_train_online(
    lesson: str,
    evaluate_inputs: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_results: List[List[ExpectationClassifierResult]],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    _test_train_online(
        lesson,
        evaluate_inputs,
        expected_training_result,
        expected_evaluate_results,
        data_root,
        shared_root,
        tmpdir,
    )
@responses.activate
@pytest.mark.parametrize(
    "lesson,evaluate_input,expected_training_result,expected_evaluate_result",
    [
        (
            "question1-with-unknown-props-in-config",
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
    evaluate_input: str,
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_result: List[_TestExpectation],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    _test_train_online(
        lesson,
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
    output_dir, accuracy = train_default_model(tmpdir, data_root, shared_root)
    assert path.exists(output_dir)
    assert round(accuracy, 2) == 0.72
    config_data = {
        "question": "What are the challenges to demonstrating integrity in a group?",
        "expectations": [
            {"ideal": "Peer pressure can cause you to allow inappropriate behavior"}
        ],
    }
    model_root, model_name = path.split(output_dir)
    classifier = SVMAnswerClassifier(model_name, [model_root], shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence="peer pressure can change your behavior",
            config_data=dict_to_config(config_data),
            expectation=0,
        )
    )
    assert len(result.expectation_results) == 1
    assert result.expectation_results[0].evaluation == "Bad"
    assert round(result.expectation_results[0].score, 2) == 0.0
