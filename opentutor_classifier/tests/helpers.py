#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from dataclasses import dataclass
from enum import Enum
import json
from os import environ, path
from typing import List, Tuple

import responses

from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ExpectationClassifierResult,
    ExpectationTrainingResult,
)
from opentutor_classifier.api import GRAPHQL_ENDPOINT
from opentutor_classifier.svm.predict import SVMAnswerClassifier


class ComparisonType(Enum):
    EQ = 0
    GTE = 1
    LTE = 2


@dataclass
class _TestExpectationClassifierResult(ExpectationClassifierResult):
    comparison: ComparisonType = ComparisonType.GTE
    epsilon: float = 0.01  # used only for eq


@dataclass
class _TestClassifierExample:
    input: AnswerClassifierInput
    expected_result: List[_TestExpectationClassifierResult]


def add_graphql_response(name: str):
    if environ.get("MOCKING_DISABLED"):
        responses.add_passthru(GRAPHQL_ENDPOINT)
        return
    with open(fixture_path(path.join("graphql", f"{name}.json"))) as f:
        responses.add(responses.POST, GRAPHQL_ENDPOINT, json=json.load(f), status=200)


def assert_train_expectation_results(
    observed: List[ExpectationTrainingResult], expected: List[ExpectationTrainingResult]
):
    for o, e in zip(observed, expected):
        assert o.accuracy >= e.accuracy


def assert_classifier_evaluate(
    observed: AnswerClassifierResult, expected: List[_TestExpectationClassifierResult]
):
    assert len(observed.expectation_results) == len(expected)
    for i in range(len(expected)):
        assert (
            observed.expectation_results[i].expectation == expected[i].expectation
            if expected[i].expectation != -1
            else i
        )
        if expected[i].comparison == ComparisonType.GTE:
            assert (
                observed.expectation_results[i].score >= expected[i].score
            ), f"classifier expectation {i} score"
        elif expected[i].comparison == ComparisonType.LTE:
            assert (
                observed.expectation_results[i].score <= expected[i].score
            ), f"classifier expectation {i} score"
        else:
            assert (
                abs(observed.expectation_results[i].score - expected[i].score)
                <= expected[i].epsilon
            ), f"classifier expectation {i} score"
        assert (
            observed.expectation_results[i].evaluation == expected[i].evaluation
        ), f"{observed.expectation_results[i].evaluation} != {expected[i].evaluation} for expectation {i}"


def run_classifier_tests(
    model_path: str, shared_root: str, examples: List[_TestClassifierExample]
):
    model_root, model_name = path.split(model_path)
    classifier = SVMAnswerClassifier(
        model_name, model_roots=[model_root], shared_root=shared_root
    )
    for ex in examples:
        assert_classifier_evaluate(classifier.evaluate(ex.input), ex.expected_result)


def create_and_test_classifier(
    model_path: str,
    shared_root: str,
    evaluate_input: str,
    expected_evaluate_result: List[_TestExpectationClassifierResult],
):
    run_classifier_tests(
        model_path,
        shared_root,
        [
            _TestClassifierExample(
                input=AnswerClassifierInput(input_sentence=evaluate_input),
                expected_result=expected_evaluate_result,
            )
        ],
    )


def fixture_path(p: str) -> str:
    return path.abspath(path.join(".", "tests", "fixtures", p))


def output_and_archive_for_test(tmpdir, data_root: str) -> Tuple[str, str]:
    testdir = tmpdir.mkdir("test")
    return (
        path.join(testdir, "model_root", path.basename(path.normpath(data_root))),
        path.join(testdir, "archive"),
    )
