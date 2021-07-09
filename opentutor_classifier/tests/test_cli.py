#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
import subprocess
import re
from typing import List, Tuple

import pytest
import responses

from opentutor_classifier import ARCH_DEFAULT
from opentutor_classifier.config import confidence_threshold_default
from .types import _TestConfig
from .utils import (
    copy_test_env_to_tmp,
    create_and_test_classifier,
    fixture_path,
    mocked_data_dao,
    _TestExpectation,
)

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.fixture(autouse=True)
def python_path_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", ".")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


def capture(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return out, err, proc.returncode


def __train_model(
    tmpdir, lesson: str, shared_root: str
) -> Tuple[str, str, str, _TestConfig]:
    config = copy_test_env_to_tmp(
        tmpdir, fixture_path("data"), shared_root, lesson=lesson
    )
    command = [
        # ".venv/bin/python3.8",
        # path.expanduser("~") + "/.cache/pypoetry/virtualenvs/opentutor-classifier-D_wtV1a--py3.8",
        "bin/opentutor_classifier",
        "train",
        "--data",
        path.join(config.data_root, lesson),
        "--shared",
        config.shared_root,
        "--output",
        config.output_dir,
    ]
    out, err, exitcode = capture(command)
    return out, err, exitcode, config


@pytest.mark.parametrize(
    "lesson,no_of_expectations",
    [("question1", 3), ("question2", 1)],
)
def test_cli_outputs_models_files(tmpdir, lesson, no_of_expectations, shared_root):
    out, err, exit_code, config = __train_model(tmpdir, lesson, shared_root)
    model_root = config.output_dir
    assert exit_code == 0
    assert path.exists(
        path.join(model_root, ARCH_DEFAULT, lesson, "models_by_expectation_num.pkl")
    )
    assert path.exists(path.join(model_root, ARCH_DEFAULT, lesson, "config.yaml"))
    out_lines = out.decode("utf-8").split("\n")
    while out_lines and re.search(r"^(DEBUG|INFO|WARNING|ERROR).*", out_lines[0]):
        out_lines.pop(0)
    assert re.search(r"Models are saved at: /.+/" + lesson, out_lines[0])
    for i in range(0, no_of_expectations):
        assert re.search(
            f"Accuracy for model={i} is [0-9]+\\.[0-9]+\\.",
            out_lines[i + 1],
            flags=re.MULTILINE,
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "lesson,answer,arch,expected_results",
    [
        (
            "question1",
            "peer pressure can change your behavior",
            ARCH_DEFAULT,
            [
                _TestExpectation(expectation=0, score=0.8, evaluation="Good"),
                # _TestExpectation(
                #     expectation=1,
                #     score=CONFIDENCE_THRESHOLD_DEFAULT,
                #     comparison=ComparisonType.LT,
                # ),
                # _TestExpectation(
                #     expectation=2,
                #     score=CONFIDENCE_THRESHOLD_DEFAULT,
                #     comparison=ComparisonType.LT,
                # ),
            ],
        ),
        (
            "question2",
            "Current flows in the same direction as the arrow",
            ARCH_DEFAULT,
            [_TestExpectation(expectation=0, score=0.81, evaluation="Good")],
        ),
    ],
)
@responses.activate
def test_cli_trained_models_usable_for_inference(
    lesson: str,
    answer: str,
    arch: str,
    expected_results: List[_TestExpectation],
    tmpdir,
    shared_root,
):
    _, _, _, config = __train_model(tmpdir, lesson, shared_root)
    model_root = config.output_dir
    assert path.exists(model_root)
    with mocked_data_dao(lesson, config.data_root, model_root, config.deployed_models):
        create_and_test_classifier(
            lesson, model_root, shared_root, answer, expected_results, arch=arch
        )
