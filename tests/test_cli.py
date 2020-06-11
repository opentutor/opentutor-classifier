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
    out_str = out.decode("utf-8")
    match_out = re.findall(
        r"M.+\/.+\n.+\=0.+[0-9]{2}\.[0-9]*\n.+\=1.+[0-9]{2}\.[0-9]*\n.+\=2.+[0-9]{2}\.[0-9]*\n",
        out_str,
    )
    assert out_str == match_out[0]
    assert len(out_str) == len(match_out[0])
    assert exitcode == 0
    return model_root


def test_cli_outputs_models_at_specified_model_root(tmpdir):
    model_root = __train_model(tmpdir)
    assert path.exists(path.join(model_root, "model_instances"))
    assert path.exists(path.join(model_root, "ideal_answers"))


def test_cli_trained_models_usable_for_inference(tmpdir):
    model_root = __train_model(tmpdir)
    assert os.path.exists(model_root)
    model_instances, ideal_answers = load_instances(model_root=model_root)
    classifier = SVMAnswerClassifier(model_instances, ideal_answers)
    result = classifier.evaluate(
        AnswerClassifierInput(input_sentence=["peer pressure"], expectation=-1)
    )
    assert len(result.expectationResults) == 3
    for exp_res in result.expectationResults:
        if exp_res.expectation == 0:
            assert exp_res.evaluation == "Good"
            assert exp_res.score == -0.6666666666666667
        if exp_res.expectation == 1:
            assert exp_res.evaluation == "Bad"
            assert exp_res.score == 1.0
        if exp_res.expectation == 2:
            assert exp_res.evaluation == "Bad"
            assert exp_res.score == 1.0
