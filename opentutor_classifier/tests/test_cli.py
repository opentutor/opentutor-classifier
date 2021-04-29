#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
from os import path
import subprocess
import re
from typing import List, Tuple

import pytest

from opentutor_classifier import ARCH_DEFAULT
from .utils import (
    create_and_test_classifier,
    fixture_path,
    output_and_archive_for_test,
    _TestExpectation,
)


@pytest.fixture(autouse=True)
def python_path_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", ".")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return os.path.dirname(word2vec)


def capture(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return out, err, proc.returncode


def __train_model(
    tmpdir, question_id: str, shared_root: str
) -> Tuple[str, str, str, str]:
    training_data_path = os.path.join(fixture_path("data"), question_id)
    output_dir, _ = output_and_archive_for_test(tmpdir, question_id)
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


def __sync(tmpdir, lesson: str, url: str) -> Tuple[str, str, str, str]:
    test_root = tmpdir.mkdir("test")
    output_dir = os.path.join(test_root, lesson)
    command = [
        ".venv/bin/python3.8",
        "bin/opentutor_classifier",
        "sync",
        "--lesson",
        lesson,
        "--url",
        url,
        "--output",
        test_root,
    ]
    out, err, exitcode = capture(command)
    return out, err, exitcode, output_dir


def test_cli_syncs_training_data(tmpdir):
    out, err, exit_code, output_root = __sync(
        tmpdir, "q1", fixture_path(os.path.join("graphql", "example-1.json"))
    )
    assert exit_code == 0
    assert path.exists(path.join(output_root, "training.csv"))
    assert path.exists(path.join(output_root, "config.yaml"))
    out_str = out.decode("utf-8")
    out_str = out_str.split("\n")
    assert out_str[0] == f"Data is saved at: {output_root}"


@pytest.mark.parametrize(
    "input_lesson,no_of_expectations", [("question1", 3), ("question2", 1)]
)
def test_cli_outputs_models_files(
    tmpdir, input_lesson, no_of_expectations, shared_root
):
    out, err, exit_code, model_root = __train_model(tmpdir, input_lesson, shared_root)
    assert exit_code == 0
    assert path.exists(path.join(model_root, "models_by_expectation_num.pkl"))
    assert path.exists(path.join(model_root, "config.yaml"))
    out_str = out.decode("utf-8")
    out_str = out_str.split("\n")
    assert re.search(r"Models are saved at: /.+/" + input_lesson, out_str[0])
    for i in range(0, no_of_expectations):
        assert re.search(
            f"Accuracy for model={i} is [0-9]+\\.[0-9]+\\.",
            out_str[i + 1],
            flags=re.MULTILINE,
        )


@pytest.mark.parametrize(
    "input_lesson,input_answer,arch,expected_results",
    [
        (
            "question1",
            "peer pressure can change your behavior",
            ARCH_DEFAULT,
            [
                _TestExpectation(expectation=0, score=0.98, evaluation="Good"),
                _TestExpectation(expectation=1, score=0.68, evaluation="Bad"),
                _TestExpectation(expectation=2, score=0.56, evaluation="Bad"),
            ],
        ),
        (
            "question2",
            "Current flows in the same direction as the arrow",
            ARCH_DEFAULT,
            [_TestExpectation(expectation=0, score=0.95, evaluation="Good")],
        ),
    ],
)
def test_cli_trained_models_usable_for_inference(
    input_lesson: str,
    input_answer: str,
    arch: str,
    expected_results: List[_TestExpectation],
    tmpdir,
    shared_root,
):
    _, _, _, model_root = __train_model(tmpdir, input_lesson, shared_root)
    assert os.path.exists(model_root)
    create_and_test_classifier(
        model_root, shared_root, input_answer, expected_results, arch=arch
    )
