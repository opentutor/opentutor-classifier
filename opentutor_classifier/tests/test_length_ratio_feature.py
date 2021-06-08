from os import path
import pytest

from opentutor_classifier import (
    ARCH_LR_CLASSIFIER,
    ArchLesson,
    ClassifierConfig,
    AnswerClassifierInput,
)

from opentutor_classifier.lr.constants import FEATURE_LENGTH_RATIO

from .utils import (
    fixture_path,
    test_env_isolated,
    train_classifier,
)


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


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


@pytest.mark.parametrize(
    "lesson,arch",
    [
        (
            "very_small_training_set",
            ARCH_LR_CLASSIFIER,
        )
    ],
)
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
def test_feature_length_ratio_disabled_by_default(
    lesson: str, arch: str, tmpdir, data_root: str, shared_root: str
):
    _test_feature_length_ratio_enabled(
        lesson, arch, tmpdir, data_root, shared_root, False
    )


def _train_classifier_and_get_confidence(
    lesson: str, arch: str, input_answer: str, tmpdir, data_root: str, shared_root: str
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        training_result = train_classifier(lesson, test_config)

        from opentutor_classifier.dao import find_data_dao

        data_dao = find_data_dao()
        question_config = data_dao.find_training_config(lesson)

        classifier_config = ClassifierConfig(
            data_dao,
            training_result.models,
            shared_root,
            model_roots=[test_config.output_dir],
        )
        from opentutor_classifier.classifier_dao import ClassifierDao

        classifier = ClassifierDao().find_classifier(classifier_config, arch)

        answer_classifier_result = classifier.evaluate(
            AnswerClassifierInput(
                input_sentence=input_answer,
                config_data=question_config,
                expectation=0,
            )
        )
        return answer_classifier_result.expectation_results[0].score


@pytest.mark.parametrize(
    "lesson,arch,input_answer",
    [
        ("long_ideal_answers_set", ARCH_LR_CLASSIFIER, "Mixture A"),
        ("long_ideal_answers_set", ARCH_LR_CLASSIFIER, "A"),
        ("long_ideal_answers_set", ARCH_LR_CLASSIFIER, "The answer is mixture A"),
    ],
)
def test_using_feature_length_ratio_lowers_confidence_w_long_ideal_answers(
    lesson: str,
    arch: str,
    input_answer: str,
    tmpdir,
    data_root: str,
    shared_root: str,
    monkeypatch,
):
    length_ratio_disabled_dir = tmpdir.mkdir("length_ratio_disabled")
    length_ratio_disabled_confidence = _train_classifier_and_get_confidence(
        lesson, arch, input_answer, length_ratio_disabled_dir, data_root, shared_root
    )

    length_ratio_enabled_dir = tmpdir.mkdir("length_ratio_enabled")
    monkeypatch.setenv(FEATURE_LENGTH_RATIO, "1")
    length_ratio_enabled_confidence = _train_classifier_and_get_confidence(
        lesson, arch, input_answer, length_ratio_enabled_dir, data_root, shared_root
    )

    assert length_ratio_enabled_confidence < length_ratio_disabled_confidence
