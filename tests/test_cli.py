import os
from os import path
import pytest
import subprocess
from opentutor_classifier import AnswerClassifierInput
from opentutor_classifier.svm import SVMAnswerClassifier, load_instances
from . import fixture_path
import re


@pytest.fixture(autouse=True)
def python_path_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", ".")


def capture(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return out, err, proc.returncode


def __train_model(tmpdir) -> str:
    test_root = tmpdir.mkdir("test")
    training_data = fixture_path(path.join("data", "training_data.csv"))
    model_root = os.path.join(test_root, "model_instances")
    command = [
        ".venv/bin/python3.8",
        "bin/opentutor_classifier",
        "train",
        "--data",
        training_data,
        "--output",
        model_root,
    ]
    out, err, exitcode = capture(command)
    print(f"err={err}")
    print(f"out={out}")
    return out, err, exitcode, model_root


def test_cli_outputs_models_at_specified_model_root(tmpdir):
    out, _, exit_code, model_root = __train_model(tmpdir)

    assert exit_code == 0
    assert path.exists(path.join(model_root, "model_instances"))
    assert path.exists(path.join(model_root, "ideal_answers"))
    out_str = out.decode("utf-8")
    out_str = out_str.split("\n")
    assert re.search(r"Models are saved at: /.+/model_instances", out_str[0])
    for i in range(0, 3):
        assert re.search(
            f"Accuracy for model={i} is [0-9]+.[0-9]+.",
            out_str[i + 1],
            flags=re.MULTILINE,
        )


def test_cli_trained_models_usable_for_inference(tmpdir):
    _, _, _, model_root = __train_model(tmpdir)
    assert os.path.exists(model_root)
    model_instances, ideal_answers = load_instances(model_root=model_root)
    classifier = SVMAnswerClassifier(model_instances, ideal_answers)
    result = classifier.evaluate(
        AnswerClassifierInput(input_sentence=["peer pressure"], expectation=-1)
    )
    assert len(result.expectation_results) == 3
    for exp_res in result.expectation_results:
        if exp_res.expectation == 0:
            assert exp_res.evaluation == "Good"
            assert exp_res.score == -0.6666666666666667
        if exp_res.expectation == 1:
            assert exp_res.evaluation == "Bad"
            assert exp_res.score == 1.0
        if exp_res.expectation == 2:
            assert exp_res.evaluation == "Bad"
            assert exp_res.score == 1.0
