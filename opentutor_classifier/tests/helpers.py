#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
from os import environ, path
from typing import List, Tuple

import responses

from opentutor_classifier import (
    AnswerClassifierInput,
    ExpectationClassifierResult,
    ExpectationTrainingResult,
)
from opentutor_classifier.api import GRAPHQL_ENDPOINT
from opentutor_classifier.lr.predict import LRAnswerClassifier


def add_graphql_response(name: str):
    if environ.get("MOCKING_DISABLED"):
        responses.add_passthru(GRAPHQL_ENDPOINT)
        return
    with open(fixture_path(path.join("graphql", f"{name}.json"))) as f:
        responses.add(responses.POST, GRAPHQL_ENDPOINT, json=json.load(f), status=200)


def assert_train_expectation_results(
    observed: List[ExpectationTrainingResult],
    expected: List[ExpectationTrainingResult],
    epsilon=0.01,
):
    for o, e in zip(observed, expected):
        assert o.accuracy >= e.accuracy  # change


def create_and_test_classifier(
    model_root: str,
    shared_root: str,
    evaluate_input: str,
    expected_evaluate_result: List[ExpectationClassifierResult],
):
    classifier = LRAnswerClassifier(model_root=model_root, shared_root=shared_root)
    evaluate_result = classifier.evaluate(
        AnswerClassifierInput(input_sentence=evaluate_input)
    )
    assert len(evaluate_result.expectation_results) == len(expected_evaluate_result)
    for i in range(len(expected_evaluate_result)):
        assert evaluate_result.expectation_results[i].expectation == i
        assert (
            round(evaluate_result.expectation_results[i].score, 2)
            >= expected_evaluate_result[i].score
        ), f"classifier expectation {i} score on input {evaluate_input}"

        assert (
            evaluate_result.expectation_results[i].evaluation
            == expected_evaluate_result[i].evaluation
        ), f"{evaluate_result.expectation_results[i].evaluation} != {expected_evaluate_result[i].evaluation} for expectation {i}"


def fixture_path(p: str) -> str:
    return path.abspath(path.join(".", "tests", "fixtures", p))


def output_and_archive_for_test(tmpdir, data_root: str) -> Tuple[str, str]:
    testdir = tmpdir.mkdir("test")
    return (
        path.join(testdir, "model_root", path.basename(path.normpath(data_root))),
        path.join(testdir, "archive"),
    )
