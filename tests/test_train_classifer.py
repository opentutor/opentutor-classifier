#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
from os import path
from opentutor_classifier import AnswerClassifierInput
from opentutor_classifier.svm.predict import SVMAnswerClassifier
from opentutor_classifier.svm.train import (
    train_classifier,
    train_classifier_online,
    train_default_classifier,
)
from opentutor_classifier.svm.utils import load_question_config
import pytest
from typing import Dict, Tuple
from . import fixture_path


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root() -> str:
    return fixture_path("shared")


def __output_dir_for_test(tmpdir, data_root: str) -> str:
    return os.path.join(
        tmpdir.mkdir("test"),
        "model_root",
        os.path.basename(os.path.normpath(data_root)),
    )


def __train_model(
    tmpdir, data_root: str, shared_root: str
) -> Tuple[str, Dict[int, int]]:
    output_dir = __output_dir_for_test(tmpdir, data_root)
    accuracy = train_classifier(
        data_root=data_root, shared_root=shared_root, output_dir=output_dir
    )
    return output_dir, accuracy


@pytest.mark.parametrize("input_lesson", [("question1"), ("question2")])
def test_outputs_models_at_specified_model_root_for_q1_and_q2(
    tmpdir, data_root: str, shared_root: str, input_lesson: str
):
    output_dir, _ = __train_model(
        tmpdir, path.join(data_root, input_lesson), shared_root
    )
    assert path.exists(path.join(output_dir, "models_by_expectation_num.pkl"))
    assert path.exists(path.join(output_dir, "config.yaml"))


def __train_default_model(
    tmpdir, data_root: str, shared_root: str
) -> Tuple[str, Dict[int, int]]:
    output_dir = os.path.join(
        tmpdir.mkdir("test"),
        "model_root",
        os.path.basename(os.path.normpath(data_root)),
    )
    accuracy = train_default_classifier(
        data_root=data_root, shared_root=shared_root, output_dir=output_dir
    )
    return output_dir, accuracy


def test_outputs_models_at_specified_model_root_for_default_model(
    tmpdir, data_root: str, shared_root: str
):
    output_dir, _ = __train_default_model(tmpdir, data_root, shared_root)
    assert path.exists(path.join(output_dir, "models_by_expectation_num.pkl"))


def test_trained_models_usable_for_inference(tmpdir, data_root: str, shared_root: str):
    output_dir, accuracy = __train_model(
        tmpdir, path.join(data_root, "question1"), shared_root
    )
    assert os.path.exists(output_dir)
    for model_num, acc in accuracy.items():
        if model_num == 0:
            assert acc == 80.0
        if model_num == 1:
            assert acc == 90.0
        if model_num == 2:
            assert acc == 100.0
    classifier = SVMAnswerClassifier(model_root=output_dir, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence="peer pressure can change your behavior",
            config_data=load_question_config({}),
            expectation=-1,
        )
    )
    assert len(result.expectation_results) == 3
    for exp_res in result.expectation_results:
        if exp_res.expectation == 0:
            assert exp_res.evaluation == "Good"
            assert round(exp_res.score, 2) == 0.99
        if exp_res.expectation == 1:
            assert exp_res.evaluation == "Bad"
            assert round(exp_res.score, 2) == 0.50
        if exp_res.expectation == 2:
            assert exp_res.evaluation == "Bad"
            assert round(exp_res.score, 2) == 0.57


def test_train_online(tmpdir, data_root: str, shared_root: str):
    lesson = "question1"
    output_dir = __output_dir_for_test(tmpdir, path.join(data_root, "question1"))
    result = train_classifier_online(
        lesson, shared_root=shared_root, output_dir=output_dir
    )
    assert result.to_dict() == {"lesson": lesson, "expectations": []}
    # assert os.path.exists(output_dir)
    # for model_num, acc in accuracy.items():
    #     if model_num == 0:
    #         assert acc == 80.0
    #     if model_num == 1:
    #         assert acc == 90.0
    #     if model_num == 2:
    #         assert acc == 100.0
    # classifier = SVMAnswerClassifier(model_root=output_dir, shared_root=shared_root)
    # result = classifier.evaluate(
    #     AnswerClassifierInput(
    #         input_sentence="peer pressure can change your behavior",
    #         config_data=load_config_into_objects({}),
    #         expectation=-1,
    #     )
    # )
    # assert len(result.expectation_results) == 3
    # for exp_res in result.expectation_results:
    #     if exp_res.expectation == 0:
    #         assert exp_res.evaluation == "Good"
    #         assert round(exp_res.score, 2) == 0.99
    #     if exp_res.expectation == 1:
    #         assert exp_res.evaluation == "Bad"
    #         assert round(exp_res.score, 2) == 0.50
    #     if exp_res.expectation == 2:
    #         assert exp_res.evaluation == "Bad"
    #         assert round(exp_res.score, 2) == 0.57


def test_trained_models_usable_for_inference_for_q2(
    tmpdir, data_root: str, shared_root: str
):
    output_dir, accuracy = __train_model(
        tmpdir, path.join(data_root, "question2"), shared_root
    )
    assert os.path.exists(output_dir)
    for model_num, acc in accuracy.items():
        if model_num == 0:
            assert acc == 100.0

    classifier = SVMAnswerClassifier(model_root=output_dir, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence="Current flows in the same direction as the arrow",
            config_data=load_question_config({}),
            expectation=0,
        )
    )
    assert len(result.expectation_results) == 1
    for exp_res in result.expectation_results:
        if exp_res.expectation == 0:
            assert exp_res.evaluation == "Good"
            assert round(exp_res.score, 2) == 0.96


def test_trained_default_model_usable_for_inference(
    tmpdir, data_root: str, shared_root: str
):
    output_dir, accuracy = __train_default_model(tmpdir, data_root, shared_root)
    assert os.path.exists(output_dir)
    assert accuracy == 72.5
    config_data = {
        "question": "What are the challenges to demonstrating integrity in a group?",
        "expectations": [
            {"ideal": "Peer pressure can cause you to allow inappropriate behavior"}
        ],
    }
    classifier = SVMAnswerClassifier(model_root=output_dir, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence="peer pressure can change your behavior",
            config_data=load_question_config(config_data),
            expectation=0,
        )
    )
    assert len(result.expectation_results) == 1
    assert result.expectation_results[0].evaluation == "Bad"
    assert round(result.expectation_results[0].score, 2) == 0.0
