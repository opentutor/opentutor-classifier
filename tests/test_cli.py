import os
from os import path
import pytest
import subprocess
from opentutor_classifier import AnswerClassifierInput
from opentutor_classifier.svm import SVMAnswerClassifier
from . import fixture_path
import re


@pytest.fixture(autouse=True)
def python_path_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", ".")


def capture(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return out, err, proc.returncode


def __train_model(tmpdir, question_id: str, shared_root: str) -> str:
    test_root = tmpdir.mkdir("test")
    training_data_path = os.path.join(fixture_path("data"), question_id)
    output_dir = os.path.join(test_root, "output", question_id)
    command = [
        ".venv/bin/python3.8",
        "bin/opentutor_classifier",
        "train",
        "--data",
        training_data_path,
        "--shared",
        shared_root,
        "--output",
        output_dir,
    ]
    out, err, exitcode = capture(command)
    return out, err, exitcode, output_dir


def test_cli_outputs_models_at_specified_model_root_for_q1(tmpdir):
    shared_root = fixture_path("shared")
    out, err, exit_code, model_root = __train_model(tmpdir, "question1", shared_root)
    assert exit_code == 0
    assert path.exists(path.join(model_root, "models_by_expectation_num.pkl"))
    assert path.exists(path.join(model_root, "ideal_answers_by_expectation_num.pkl"))
    assert path.exists(path.join(model_root, "config.yaml"))
    out_str = out.decode("utf-8")
    out_str = out_str.split("\n")
    assert re.search(r"Models are saved at: /.+/output", out_str[0])
    for i in range(0, 3):
        assert re.search(
            f"Accuracy for model={i} is [0-9]+\\.[0-9]+\\.",
            out_str[i + 1],
            flags=re.MULTILINE,
        )


def test_cli_outputs_models_at_specified_model_root_for_q2(tmpdir):
    shared_root = fixture_path("shared")
    out, err, exit_code, model_root = __train_model(tmpdir, "question2", shared_root)
    assert exit_code == 0
    assert path.exists(path.join(model_root, "models_by_expectation_num.pkl"))
    assert path.exists(path.join(model_root, "ideal_answers_by_expectation_num.pkl"))
    assert path.exists(path.join(model_root, "config.yaml"))
    out_str = out.decode("utf-8")
    out_str = out_str.split("\n")
    assert re.search(r"Models are saved at: /.+/output", out_str[0])
    for i in range(0, 1):
        assert re.search(
            f"Accuracy for model={i} is [0-9]+\\.[0-9]+\\.",
            out_str[i + 1],
            flags=re.MULTILINE,
        )


def test_cli_trained_models_usable_for_inference_for_q1(tmpdir):
    shared_root = fixture_path("shared")
    _, _, _, model_root = __train_model(tmpdir, "question1", shared_root)
    assert os.path.exists(model_root)
    classifier = SVMAnswerClassifier(model_root=model_root, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence="peer pressure can change your behavior", expectation=-1
        )
    )
    assert len(result.expectation_results) == 3
    for exp_res in result.expectation_results:
        if exp_res.expectation == 0:
            assert exp_res.evaluation == "Good"
            assert round(exp_res.score, 2) == 0.94
        if exp_res.expectation == 1:
            assert exp_res.evaluation == "Bad"
            assert round(exp_res.score, 2) == 0.23
        if exp_res.expectation == 2:
            assert exp_res.evaluation == "Bad"
            assert round(exp_res.score, 2) == 0.28


def test_cli_trained_models_usable_for_inference_for_q2(tmpdir):
    shared_root = fixture_path("shared")
    _, _, _, model_root = __train_model(tmpdir, "question2", shared_root)
    assert os.path.exists(model_root)
    classifier = SVMAnswerClassifier(model_root=model_root, shared_root=shared_root)
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
            assert round(exp_res.score, 2) == 0.93
