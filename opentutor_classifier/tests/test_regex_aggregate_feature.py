#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
import pytest

from opentutor_classifier import (
    ARCH_SVM_CLASSIFIER,
    ArchLesson,
)

from opentutor_classifier.svm.constants import FEATURE_REGEX_AGGREGATE_ENABLED
from opentutor_classifier.config import confidence_threshold_default

from .utils import (
    fixture_path,
    test_env_isolated,
    train_classifier,
    read_example_testset,
    run_classifier_testset,
    assert_testset_accuracy,
)


CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


def _test_regex_aggregate_enabled(
    lesson: str,
    arch: str,
    tmpdir,
    data_root: str,
    shared_root: str,
    expect_enabled: bool,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        train_classifier(lesson, test_config)
        from opentutor_classifier.dao import find_data_dao

        dao = find_data_dao()
        pconfig = dao.find_prediction_config(ArchLesson(arch=arch, lesson=lesson))
        assert (
            bool(pconfig.expectations[0].features[FEATURE_REGEX_AGGREGATE_ENABLED])
            == expect_enabled
        )


@pytest.mark.parametrize(
    "lesson,arch",
    [
        (
            "proportion",
            ARCH_SVM_CLASSIFIER,
        )
    ],
)
def test_regex_aggregate_can_be_enabled_w_env_var(
    lesson: str,
    arch: str,
    tmpdir,
    data_root: str,
    shared_root: str,
    monkeypatch,
):
    monkeypatch.setenv(FEATURE_REGEX_AGGREGATE_ENABLED, "1")
    _test_regex_aggregate_enabled(lesson, arch, tmpdir, data_root, shared_root, True)


@pytest.mark.parametrize(
    "lesson,arch",
    [
        (
            "proportion",
            ARCH_SVM_CLASSIFIER,
        )
    ],
)
def test_feature_regex_aggregate_disabled_by_default(
    lesson: str, arch: str, tmpdir, data_root: str, shared_root: str
):
    _test_regex_aggregate_enabled(lesson, arch, tmpdir, data_root, shared_root, False)


@pytest.mark.parametrize(
    "lesson,arch,confidence_threshold,feature_env_var_enabled_at_train_time,feature_env_var_enabled_at_predict_time",
    [
        ("proportion", ARCH_SVM_CLASSIFIER, CONFIDENCE_THRESHOLD_DEFAULT, True, True),
    ],
)
def test_classifier_and_get_accuracy(
    lesson: str,
    arch: str,
    feature_env_var_enabled_at_train_time: bool,
    feature_env_var_enabled_at_predict_time: bool,
    tmpdir,
    data_root: str,
    shared_root: str,
    monkeypatch,
    confidence_threshold: float,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        monkeypatch.setenv(
            FEATURE_REGEX_AGGREGATE_ENABLED, str(feature_env_var_enabled_at_train_time)
        )
        enabled_result = train_classifier(lesson, test_config)
        assert path.exists(enabled_result.models)
        monkeypatch.setenv(
            FEATURE_REGEX_AGGREGATE_ENABLED,
            str(feature_env_var_enabled_at_predict_time),
        )
        disabled_result = train_classifier(lesson, test_config)
        assert path.exists(disabled_result.models)
        testset = read_example_testset(
            lesson, confidence_threshold=confidence_threshold
        )
        result = run_classifier_testset(
            arch, disabled_result.models, shared_root, testset
        )
        metrics = result.metrics()
        assert_testset_accuracy(
            arch,
            enabled_result.models,
            shared_root,
            testset,
            expected_accuracy=metrics.accuracy,
        )
