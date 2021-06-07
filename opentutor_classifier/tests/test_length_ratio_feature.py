from os import path

import pytest

from opentutor_classifier import ARCH_LR_CLASSIFIER
from opentutor_classifier.config import confidence_threshold_default
from .utils import (
    assert_length_ratio_feature_toggle_consistent,
    create_and_test_classifier,
    fixture_path,
    read_example_testset,
    test_env_isolated,
    train_classifier,
)

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()

@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)

@pytest.fixture
def length_ratio_feature_toggle(monkeypatch, use_length_ratio):
    monkeypatch.setenv("USE_LENGTH_RATIO", use_length_ratio)
    return use_length_ratio


def _test_train_and_predict_length_ratio_consistent(
    lesson: str, # ideal answers long dataset
    arch: str,
    # confidence_threshold for now determines whether an answer
    # is really classified as GOOD/BAD (confidence >= threshold)
    # or whether it is interpretted as NEUTRAL (confidence < threshold)
    confidence_threshold: float,
    # expected_training_result: List[ExpectationTrainingResult],
    # expected_accuracy: float,
    tmpdir,
    data_root: str,
    shared_root: str,
    use_length_ratio: str,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)
        config_path = path.join(fixture_path('models'), )
        # call the config check twice? before training and after prediction?
        assert_length_ratio_feature_toggle_consistent(
            test_config.output_dir, lesson, use_length_ratio
        )
        testset = read_example_testset(
            lesson, confidence_threshold=confidence_threshold
        )
        # assert_length_ratio_feature_toggle_consistent(
        #     config_path, use_length_ratio
        # )
        assert_length_ratio_feature_toggle_consistent(
            test_config.output_dir, lesson, use_length_ratio
        )

@pytest.mark.parametrize(
    "example,arch,confidence_threshold,use_length_ratio",
    [
        (
            # "long-ideal-answers",
            "ies-rectangle",
            ARCH_LR_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            "True",
        ),
        (
            # "long-ideal-answers",
            "ies-rectangle",
            ARCH_LR_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            "False",
        ),
    ],
)

@pytest.mark.only
def test_can_toggle_length_ratio_feature(
    example: str,
    arch: str,
    # confidence_threshold for now determines whether an answer
    # is really classified as GOOD/BAD (confidence >= threshold)
    # or whether it is interpretted as NEUTRAL (confidence < threshold)
    confidence_threshold: float,
    tmpdir,
    data_root: str,
    shared_root: str,
    length_ratio_feature_toggle
):
    _test_train_and_predict_length_ratio_consistent(
        example,
        arch,
        confidence_threshold,
        tmpdir,
        data_root,
        shared_root,
        length_ratio_feature_toggle
    )