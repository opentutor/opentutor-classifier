import os
from os import path
from opentutor_classifier import AnswerClassifierInput
from opentutor_classifier.svm import train_classifier, SVMAnswerClassifier
import pytest
from typing import Dict, Tuple
from . import fixture_path


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root() -> str:
    return fixture_path("shared")


def __train_model(
    tmpdir, data_root: str, shared_root: str
) -> Tuple[str, Dict[int, int]]:
    output_dir = os.path.join(
        tmpdir.mkdir("test"),
        "model_root",
        os.path.basename(os.path.normpath(data_root)),
    )
    accuracy = train_classifier(
        data_root=data_root, shared_root=shared_root, output_dir=output_dir
    )
    return output_dir, accuracy


def test_outputs_models_at_specified_model_root_for_q1(
    tmpdir, data_root: str, shared_root: str
):
    output_dir, _ = __train_model(
        tmpdir, path.join(data_root, "question1"), shared_root
    )
    assert path.exists(path.join(output_dir, "models_by_expectation_num.pkl"))
    assert path.exists(path.join(output_dir, "ideal_answers_by_expectation_num.pkl"))
    assert path.exists(path.join(output_dir, "config.yaml"))


def test_outputs_models_at_specified_model_root_for_q2(
    tmpdir, data_root: str, shared_root: str
):
    output_dir, _ = __train_model(
        tmpdir, path.join(data_root, "question2"), shared_root
    )
    assert path.exists(path.join(output_dir, "models_by_expectation_num.pkl"))
    assert path.exists(path.join(output_dir, "ideal_answers_by_expectation_num.pkl"))
    assert path.exists(path.join(output_dir, "config.yaml"))


def test_trained_models_usable_for_inference(tmpdir, data_root: str, shared_root: str):
    output_dir, accuracy = __train_model(
        tmpdir, path.join(data_root, "question1"), shared_root
    )
    assert os.path.exists(output_dir)
    for model_num, acc in accuracy.items():
        if model_num == 0:
            assert acc == 80.0
        if model_num == 1:
            assert acc == 80.0
        if model_num == 2:
            assert acc == 90.0
    classifier = SVMAnswerClassifier(model_root=output_dir, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence="peer pressure can change your behavior", expectation=-1
        )
    )
    assert len(result.expectation_results) == 3
    for exp_res in result.expectation_results:
        if exp_res.expectation == 0:
            assert exp_res.evaluation == "Good"
            assert round(exp_res.score, 2) == 0.93
        if exp_res.expectation == 1:
            assert exp_res.evaluation == "Good"
            assert round(exp_res.score, 2) == 0.89
        if exp_res.expectation == 2:
            assert exp_res.evaluation == "Bad"
            assert round(exp_res.score, 2) == 0.16


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
            expectation=0,
        )
    )
    assert len(result.expectation_results) == 1
    for exp_res in result.expectation_results:
        if exp_res.expectation == 0:
            assert exp_res.evaluation == "Good"
            assert round(exp_res.score, 2) == 0.96
