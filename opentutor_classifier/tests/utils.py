#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from contextlib import contextmanager
from distutils.dir_util import copy_tree
import logging
from pathlib import Path
from os import path
from typing import List, Tuple
from unittest.mock import patch

import pandas as pd
import pytest
import responses


from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ClassifierFactory,
    ClassifierConfig,
    DataDao,
    ExpectationClassifierResult,
    ExpectationTrainingResult,
    FeaturesSaveRequest,
    QuestionConfig,
    TrainingConfig,
    TrainingInput,
    TrainingOptions,
    TrainingResult,
)
from opentutor_classifier.training import train_data_root, train_default
from opentutor_classifier.api import FileDataDao, get_graphql_endpoint
from opentutor_classifier.config import (
    LABEL_BAD,
    LABEL_GOOD,
    LABEL_NEUTRAL,
    confidence_threshold_default,
)
from .types import (
    ComparisonType,
    _TestConfig,
    _TestExample,
    _TestExampleResult,
    _TestExpectation,
    _TestExpectationResult,
    _TestSet,
    _TestSetResult,
)


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
    if expected.evaluation and effective_evaluation != expected.evaluation.lower():
        errors.append(
            f"expected evaluation '{observed.evaluation}' to be '{expected.evaluation}'"
        )
    return _TestExpectationResult(expected=expected, observed=observed, errors=errors)


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
    lesson: str,
    arch: str,
    model_root: str,
    shared_root: str,
    examples: List[_TestExample],
):
    from opentutor_classifier.log import logger

    logger.warning(f"\n\nrun_classifier_tests model_root={model_root}")
    classifier = ClassifierFactory().new_classifier(
        ClassifierConfig(
            model_name=lesson, model_roots=[model_root], shared_root=shared_root
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
    lesson: str,
    model_root: str,
    shared_root: str,
    evaluate_input: str,
    expected_evaluate_result: List[_TestExpectation],
    arch: str = "",
):
    run_classifier_tests(
        lesson,
        arch,
        model_root,
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


def copy_test_env_to_tmp(
    tmpdir,
    data_root: str,
    shared_root: str,
    arch="",
    lesson="",
    is_default_model: bool = False,
) -> _TestConfig:
    testdir = tmpdir.mkdir("test")
    config = _TestConfig(
        arch=arch,
        archive_root=path.join(testdir, "archive"),
        data_root=path.join(testdir, "data"),
        is_default_model=is_default_model,
        output_dir=path.join(
            testdir, "model_root", path.basename(path.normpath(data_root))
        ),
        shared_root=shared_root,
    )
    copy_tree(path.join(data_root, lesson), path.join(config.data_root, lesson))
    return config


def _add_gql_response(config: _TestConfig, lesson: str):
    cfile = Path(path.join(config.data_root, lesson, "config.yaml"))
    dfile = Path(path.join(config.data_root, lesson, "training.csv"))
    training_data_prop = (
        "allTrainingData" if config.is_default_model else "trainingData"
    )
    res = {
        "data": {
            "me": {
                training_data_prop: {
                    "config": cfile.read_text() if cfile.is_file() else None,
                    "training": dfile.read_text() if dfile.is_file() else None,
                }
            }
        }
    }
    responses.add(responses.POST, get_graphql_endpoint(), json=res, status=200)


class _TestDataDao(DataDao):
    """
    Wrapper DataDao for tests.
    We need this because if the underlying DataDao is Gql,
    then after DataDao::save_features is called,
    we need to add a new mocked graphql response with the updated features
    """

    def __init__(self, dao: FileDataDao, config: _TestConfig):
        self.dao = dao
        self.config = config

    def find_config(self, lesson: str) -> QuestionConfig:
        return self.dao.find_config(lesson)

    def find_training_input(self, lesson: str) -> TrainingInput:
        return self.dao.find_training_input(lesson)

    def save_features(self, req: FeaturesSaveRequest) -> None:
        self.dao.save_features(req)
        _add_gql_response(self.config, req.lesson)


@contextmanager
def test_env_isolated(
    tmpdir,
    data_root: str,
    shared_root: str,
    arch="",
    lesson="",
    is_default_model=False,
):
    config = copy_test_env_to_tmp(
        tmpdir,
        data_root,
        shared_root,
        arch=arch,
        is_default_model=is_default_model,
        lesson=lesson,
    )
    _add_gql_response(config, lesson)
    patcher = patch("opentutor_classifier.find_data_dao")
    try:
        mock_find_data_dao = patcher.start()
        mock_find_data_dao.return_value = _TestDataDao(
            FileDataDao(config.data_root), config
        )
        yield config
    finally:
        patcher.stop()


def train_classifier(lesson: str, config: _TestConfig) -> TrainingResult:
    return train_data_root(
        data_root=path.join(config.data_root, lesson),
        config=TrainingConfig(shared_root=config.shared_root),
        opts=TrainingOptions(
            archive_root=config.archive_root,
            output_dir=config.output_dir,
        ),
        arch=config.arch,
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
    for _, row in df.iterrows():
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
