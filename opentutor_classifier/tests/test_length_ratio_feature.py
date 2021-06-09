from os import path

import pytest

from opentutor_classifier import ARCH_LR_CLASSIFIER, ArchLesson

# from opentutor_classifier.config import confidence_threshold_default
# from opentutor_classifier.utils import load_yaml
from opentutor_classifier.lr.constants import FEATURE_LENGTH_RATIO

from .utils import (
    fixture_path,
    # read_example_testset,
    test_env_isolated,
    train_classifier,
)


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


# def _test_train_and_predict_length_ratio_consistent(
#     lesson: str,  # ideal answers long dataset
#     arch: str,
#     # confidence_threshold for now determines whether an answer
#     # is really classified as GOOD/BAD (confidence >= threshold)
#     # or whether it is interpretted as NEUTRAL (confidence < threshold)
#     confidence_threshold: float,
#     # expected_training_result: List[ExpectationTrainingResult],
#     # expected_accuracy: float,
#     tmpdir,
#     data_root: str,
#     shared_root: str,
#     use_length_ratio: str,
# ):
#     with test_env_isolated(
#         tmpdir, data_root, shared_root, arch=arch, lesson=lesson
#     ) as test_config:
#         train_result = train_classifier(lesson, test_config)
#         assert path.exists(train_result.models)
#         config_path = path.join(
#             fixture_path("models"),
#         )
#         # call the config check twice? before training and after prediction?
#         assert_length_ratio_feature_toggle_consistent(
#             test_config.output_dir, lesson, use_length_ratio
#         )
#         testset = read_example_testset(
#             lesson, confidence_threshold=confidence_threshold
#         )
#         # assert_length_ratio_feature_toggle_consistent(
#         #     config_path, use_length_ratio
#         # )
#         assert_length_ratio_feature_toggle_consistent(
#             test_config.output_dir, lesson, use_length_ratio
#         )


def _test_feature_length_ratio_enabled(
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
            bool(pconfig.expectations[0].features[FEATURE_LENGTH_RATIO])
            == expect_enabled
        )


@pytest.mark.only
@pytest.mark.parametrize(
    "lesson,arch",
    [
        (
            "very_small_training_set",
            ARCH_LR_CLASSIFIER,
        )
    ],
)
@pytest.mark.only
def test_feature_length_ratio_can_be_enabled_w_env_var(
    lesson: str,
    arch: str,
    tmpdir,
    data_root: str,
    shared_root: str,
    monkeypatch,
):
    monkeypatch.setenv(FEATURE_LENGTH_RATIO, "1")
    _test_feature_length_ratio_enabled(
        lesson, arch, tmpdir, data_root, shared_root, True
    )


@pytest.mark.parametrize(
    "lesson,arch",
    [
        (
            "very_small_training_set",
            ARCH_LR_CLASSIFIER,
        )
    ],
)
@pytest.mark.only
def test_feature_length_ratio_disabled_by_default(
    lesson: str, arch: str, tmpdir, data_root: str, shared_root: str
):
    _test_feature_length_ratio_enabled(
        lesson, arch, tmpdir, data_root, shared_root, False
    )
