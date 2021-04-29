#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import logging
from os import environ, path
from typing import List, Tuple

import pandas as pd
import pytest
import responses

from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ClassifierFactory,
    ClassifierConfig,
    ExpectationClassifierResult,
    ExpectationTrainingResult,
    TrainingConfig,
    TrainingOptions,
    TrainingResult,
)
from opentutor_classifier.training import train_data_root, train_default
from opentutor_classifier.api import GRAPHQL_ENDPOINT
from opentutor_classifier.config import (
    LABEL_BAD,
    LABEL_GOOD,
    LABEL_NEUTRAL,
    confidence_threshold_default,
)
from .types import (
    ComparisonType,
    _TestExample,
    _TestExampleResult,
    _TestExpectation,
    _TestExpectationResult,
    _TestSet,
    _TestSetResult,
)


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


def to_expectation_result(
    expected: _TestExpectation, observed: ExpectationClassifierResult
) -> _TestExpectationResult:
    errors = []
    effective_evaluation = observed.evaluation.lower()
    # if observed.expectation != expected.expectation:
    #     errors.append(f"expected result expectation index {observed.expectation} to be {expected.expectation}")
    if expected.comparison == ComparisonType.GTE:
        if observed.score < expected.score:
            errors.append(f"expected score {observed.score} to be >= {expected.score}")
    elif expected.comparison == ComparisonType.LT:
        if observed.score >= expected.score:
            errors.append(f"expected score {observed.score} to be < {expected.score}")
        else:
            effective_evaluation = (
                LABEL_NEUTRAL  # hack until we fix the way classifier reports neutral
            )
    else:
        if abs(observed.score - expected.score) > expected.epsilon:
            errors.append(
                f"expected score {observed.score} to be within {expected.epsilon} of {expected.score} (observed difference {observed.score - expected.score}"
            )
    if effective_evaluation != expected.evaluation.lower():
        errors.append(
            f"expected evaluation '{observed.evaluation}' to be '{expected.evaluation}'"
        )
    return _TestExpectationResult(expected=expected, observed=observed, errors=errors)


def to_example_result(
    expected: _TestExample, observed: AnswerClassifierResult
) -> _TestExampleResult:
    return _TestExampleResult(
        expected=expected,
        observed=observed,
        expectations=[
            to_expectation_result(e, o)
            for e, o in zip(expected.expectations, observed.expectation_results)
        ],
    )


def assert_classifier_evaluate(
    observed: AnswerClassifierResult, expected: List[_TestExpectation]
):
    result = to_example_result(
        _TestExample(input=observed.input, expectations=expected),
        observed,
    )
    if result.is_failure():
        pytest.fail(result.errors())


def run_classifier_tests(
    arch: str, model_path: str, shared_root: str, examples: List[_TestExample]
):
    model_root, model_name = path.split(model_path)
    classifier = ClassifierFactory().new_classifier(
        ClassifierConfig(
            model_name=model_name, model_roots=[model_root], shared_root=shared_root
        ),
        arch=arch,
    )
    for ex in examples:
        assert_classifier_evaluate(classifier.evaluate(ex.input), ex.expectations)


def run_classifier_testset(
    arch: str, model_path: str, shared_root: str, testset: _TestSet
) -> _TestSetResult:
    model_root, model_name = path.split(model_path)
    classifier = ClassifierFactory().new_classifier(
        ClassifierConfig(
            model_name=model_name, model_roots=[model_root], shared_root=shared_root
        ),
        arch=arch,
    )
    result = _TestSetResult(testset=testset)
    for ex in testset.examples:
        result.results.append(to_example_result(ex, classifier.evaluate(ex.input)))
    return result


def assert_testset_accuracy(
    arch: str,
    model_path: str,
    shared_root: str,
    testset: _TestSet,
    expected_accuracy=1.0,
) -> None:
    result = run_classifier_testset(arch, model_path, shared_root, testset)
    metrics = result.metrics()
    if metrics.accuracy >= expected_accuracy:
        return
    logging.warning("ERRORS:\n" + "\n".join(ex.errors() for ex in result.results))
    assert metrics.accuracy >= expected_accuracy


def create_and_test_classifier(
    model_path: str,
    shared_root: str,
    evaluate_input: str,
    expected_evaluate_result: List[_TestExpectation],
    arch: str = "",
):
    run_classifier_tests(
        arch,
        model_path,
        shared_root,
        [
            _TestExample(
                input=AnswerClassifierInput(input_sentence=evaluate_input),
                expectations=expected_evaluate_result,
            )
        ],
    )


def fixture_path(p: str) -> str:
    return path.abspath(path.join(".", "tests", "fixtures", p))


def example_testset_path(example: str, testset_name="test.csv") -> str:
    return fixture_path(path.join("data", example, testset_name))


def output_and_archive_for_test(tmpdir, data_root: str) -> Tuple[str, str]:
    testdir = tmpdir.mkdir("test")
    return (
        path.join(testdir, "model_root", path.basename(path.normpath(data_root))),
        path.join(testdir, "archive"),
    )


def train_classifier(
    tmpdir, data_root: str, shared_root: str, arch=""
) -> TrainingResult:
    output_dir, archive_root = output_and_archive_for_test(tmpdir, data_root)
    return train_data_root(
        data_root=data_root,
        config=TrainingConfig(shared_root=shared_root),
        opts=TrainingOptions(
            archive_root=archive_root,
            output_dir=output_dir,
        ),
        arch=arch,
    )


def train_default_model(
    tmpdir, data_root: str, shared_root: str, arch=""
) -> Tuple[str, TrainingResult]:
    output_dir = path.join(
        tmpdir.mkdir("test"), "model_root", path.basename(path.normpath(data_root))
    )
    result = train_default(
        data_root=data_root,
        arch=arch,
        config=TrainingConfig(shared_root=shared_root),
        opts=TrainingOptions(output_dir=output_dir),
    )
    return output_dir, result


def read_test_set_from_csv(csv_path: str, confidence_threshold=-1.0) -> _TestSet:
    confidence_threshold_effective = (
        confidence_threshold
        if confidence_threshold >= 0.0
        else confidence_threshold_default()
    )
    df = pd.read_csv(csv_path, header=0)
    df.fillna("", inplace=True)
    testset = _TestSet()
    for i, row in df.iterrows():
        testset.examples.append(
            _TestExample(
                input=AnswerClassifierInput(expectation=row[0], input_sentence=row[1]),
                expectations=[
                    _TestExpectation(
                        expectation=row[0],
                        evaluation=row[2],
                        score=confidence_threshold_effective,
                        comparison=ComparisonType.GTE
                        if row[2] in [LABEL_BAD, LABEL_GOOD]
                        else ComparisonType.LT,
                    )
                ],
            )
        )
    return testset


def read_example_testset(
    example: str, confidence_threshold=-1.0, testset_name="test.csv"
) -> _TestSet:
    return read_test_set_from_csv(
        example_testset_path(
            example,
            testset_name=testset_name,
        ),
        confidence_threshold=confidence_threshold,
    )
