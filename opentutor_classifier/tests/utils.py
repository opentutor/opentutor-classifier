#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
from contextlib import contextmanager
from distutils.dir_util import copy_tree
import logging
from pathlib import Path
from typing import Any, List
from unittest.mock import patch

import pandas as pd
import pytest
import responses

from opentutor_classifier.training import train_default_data_root


from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ArchLesson,
    ClassifierFactory,
    ClassifierConfig,
    ExpectationClassifierResult,
    ExpectationTrainingResult,
    QuestionConfig,
    QuestionConfigSaveReq,
    ModelRef,
    ModelSaveReq,
    TrainingConfig,
    TrainingInput,
    TrainingResult,
)
from opentutor_classifier.training import train_data_root
from opentutor_classifier.api import get_graphql_endpoint
from opentutor_classifier.config import (
    LABEL_BAD,
    LABEL_GOOD,
    LABEL_NEUTRAL,
    confidence_threshold_default,
)
from opentutor_classifier import DataDao
import opentutor_classifier.dao
from opentutor_classifier.dao import FileDataDao
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
    classifier = ClassifierFactory().new_classifier(
        ClassifierConfig(
            dao=opentutor_classifier.dao.find_data_dao(),
            model_name=lesson,
            model_roots=[model_root],
            shared_root=shared_root,
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
            dao=opentutor_classifier.dao.find_data_dao(),
            model_name=model_name,
            model_roots=[model_root],
            shared_root=shared_root,
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


def example_data_path(example: str) -> str:
    return fixture_path(path.join("data", example))


def example_testset_path(example: str, testset_name="test.csv") -> str:
    return path.join(example_data_path(example), testset_name)


def copy_test_env_to_tmp(
    tmpdir,
    data_root: str,
    shared_root: str,
    arch="",
    deployed_models="",
    lesson="",
    is_default_model: bool = False,
) -> _TestConfig:
    testdir = tmpdir.mkdir("test")
    config = _TestConfig(
        arch=arch,
        data_root=path.join(testdir, "data"),
        deployed_models=deployed_models or fixture_path("models_deployed"),
        is_default_model=is_default_model,
        output_dir=path.join(
            testdir, "model_root", path.basename(path.normpath(data_root))
        ),
        shared_root=shared_root,
    )
    copy_tree(path.join(data_root, lesson), path.join(config.data_root, lesson))
    return config


def mock_gql_response(lesson: str, data_root: str, is_default_model=False):
    cfile = Path(path.join(data_root, lesson, "config.yaml"))
    dfile = Path(path.join(data_root, lesson, "training.csv"))
    training_data_prop = "allTrainingData" if is_default_model else "trainingData"
    config_stringified = cfile.read_text() if cfile.is_file() else None
    res = {
        "data": {
            "me": {
                training_data_prop: {
                    "config": config_stringified,
                    "training": dfile.read_text() if dfile.is_file() else None,
                },
                "config": {
                    "stringified": config_stringified,
                },
            }
        }
    }
    responses.add(responses.POST, get_graphql_endpoint(), json=res, status=200)


class _TestDataDao(DataDao):
    """
    Wrapper DataDao for tests.
    We need this because if the underlying DataDao is Gql,
    then after DataDao::save_config is called,
    we need to add a new mocked graphql response with the updated features
    """

    def __init__(self, dao: FileDataDao, is_default_model=False):
        self.dao = dao
        self.is_default_model = is_default_model

    @property
    def data_root(self) -> str:
        return self.dao.data_root

    @property
    def model_root(self) -> str:
        return self.dao.model_root

    def get_model_root(self, lesson: ArchLesson) -> str:
        return self.dao.get_model_root(lesson)

    def find_default_training_data(self) -> pd.DataFrame:
        return self.dao.find_default_training_data()

    def find_prediction_config(self, lesson: ArchLesson) -> QuestionConfig:
        return self.dao.find_prediction_config(lesson)

    def find_training_config(self, lesson: str) -> QuestionConfig:
        return self.dao.find_training_config(lesson)

    def find_training_input(self, lesson: str) -> TrainingInput:
        return self.dao.find_training_input(lesson)

    def load_pickle(self, ref: ModelRef) -> Any:
        return self.dao.load_pickle(ref)

    def trained_model_exists(self, ref: ModelRef) -> bool:
        return self.dao.trained_model_exists(ref)

    def save_config(self, req: QuestionConfigSaveReq) -> None:
        self.dao.save_config(req)
        mock_gql_response(
            req.lesson,
            self.dao.data_root,
            is_default_model=self.is_default_model,
        )

    def save_pickle(self, req: ModelSaveReq) -> None:
        self.dao.save_pickle(req)


@contextmanager
def mocked_data_dao(
    lesson: str,
    data_root: str,
    model_root: str,
    deployed_model_root: str,
    is_default_model=False,
):
    patcher = patch("opentutor_classifier.dao.find_data_dao")
    try:
        mock_gql_response(
            lesson,
            data_root,
            is_default_model=is_default_model,
        )
        mock_find_data_dao = patcher.start()
        mock_find_data_dao.return_value = _TestDataDao(
            FileDataDao(
                data_root,
                model_root=model_root,
                deployed_model_root=deployed_model_root,
            ),
            is_default_model,
        )
        yield None
    finally:
        patcher.stop()


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
    mock_gql_response(
        lesson,
        config.data_root,
        is_default_model=config.is_default_model,
    )
    patcher = patch("opentutor_classifier.dao.find_data_dao")
    try:
        mock_find_data_dao = patcher.start()
        mock_find_data_dao.return_value = _TestDataDao(
            FileDataDao(config.data_root, model_root=config.output_dir),
            config.is_default_model,
        )
        yield config
    finally:
        patcher.stop()


def train_classifier(lesson: str, config: _TestConfig) -> TrainingResult:
    return train_data_root(
        data_root=path.join(config.data_root, lesson),
        config=TrainingConfig(shared_root=config.shared_root),
        output_dir=config.output_dir,
        arch=config.arch,
    )


def train_default_classifier(config: _TestConfig) -> TrainingResult:
    return train_default_data_root(
        data_root=path.join(config.data_root, "default"),
        config=TrainingConfig(shared_root=config.shared_root),
        output_dir=config.output_dir,
        arch=config.arch,
    )


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
    result_expectations = []
    for e in expected.expectations:
        for o in observed.expectation_results:
            if e.expectation == o.expectation:
                result_expectations.append(to_expectation_result(e, o))
    return _TestExampleResult(
        expected=expected, observed=observed, expectations=result_expectations
    )
