#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
import pytest
import responses
from typing import List, Tuple

from opentutor_classifier import (
    ARCH_LR2_CLASSIFIER,
    ModelRef,
)
from opentutor_classifier.dao import find_predicton_config_and_pickle
from opentutor_classifier.lr2.features import preprocess_sentence
from opentutor_classifier.lr2.clustering_features import CustomDBScanClustering
from opentutor_classifier.lr2.constants import MODEL_FILE_NAME
from opentutor_classifier.spacy_preprocessor import SpacyPreprocessor
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


@pytest.mark.parametrize(
    "sentence, expected_transformation",
    [
        (
            "thirty",
            [
                "30",
            ],
        ),
        ("thirty seven by forty", ["37", "40"]),
        (
            "mixture a would be proportional",
            ["mixture", "a", "would", "proportional"],
        ),
    ],
)
def test_text2num(sentence: str, expected_transformation: str, shared_root: str):
    preprocessor = SpacyPreprocessor(shared_root)
    transformed_tranform = preprocess_sentence(sentence, preprocessor)
    assert (
        expected_transformation == transformed_tranform
    ), f"Expected {expected_transformation} got {transformed_tranform}"


@pytest.mark.parametrize(
    "input_patterns_with_fpr, expected_patterns, cuttoff_fpr",
    [
        (
            [
                ("square", 0.7),
                ("rectangle", 0.90),
                ("like + rectangle", 0.98),
                ("like + square", 0.6),
            ],
            ["like + rectangle", "rectangle", "square"],
            0.6,
        ),
        (
            [("uniform", 0.8), ("burn", 0.5), ("candles + burn + uniform", 0.75)],
            ["candles + burn + uniform", "uniform"],
            0.7,
        ),
    ],
)
def test_unit_deduplication(
    input_patterns_with_fpr: List[Tuple[str, float]],
    expected_patterns: List[str],
    cuttoff_fpr: float,
):
    patterns = CustomDBScanClustering.deduplicate_patterns(
        input_patterns_with_fpr, cuttoff_fpr
    )
    assert patterns == expected_patterns, f"Expected {expected_patterns} got {patterns}"


@pytest.mark.parametrize(
    "patterns, input_x, input_y, n, expected_patterns",
    [
        (
            ["37 + 40", "37", "sides + closer", "1 + ratio", "difference + length"],
            [
                "37 by 40 is the answer",
                "37 x 40 rectangele looks more like square",
                "rectangle with sides length closer to each other",
                "more difference in sides length make it look like a square",
                "rectangle with sides ratio close to 1 looks like a square",
                "if difference in sides length is less then it is a square",
                "lesser difference in side length make it a square",
                "37 x 40 does not look like a square",
                "the one with bigger sides",
                "37 feet by 40 feet",
            ],
            [
                "good",
                "good",
                "good",
                "bad",
                "good",
                "good",
                "good",
                "bad",
                "bad",
                "good",
            ],
            4,
            ['37 + 40', '37', 'sides + closer', '1 + ratio'],
        )
    ],
)
def test_univariate_selection(
    patterns: List[str],
    input_x: List[str],
    input_y: List[str],
    n: int,
    expected_patterns: List[str],
    shared_root: str,
):
    preprocessor = SpacyPreprocessor(shared_root)
    patterns = CustomDBScanClustering.univariate_feature_selection(
        patterns, input_x, input_y, preprocessor, n
    )
    assert patterns == expected_patterns, f"Expected {expected_patterns} got {patterns}"


@responses.activate
@pytest.mark.parametrize(
    "lesson,arch,train_quality,required_fields",
    [
        (
            "shapes",
            ARCH_LR2_CLASSIFIER,
            2,
            [
                "good",
                "bad",
                "featureLengthRatio",
                "featureRegexAggregateDisabled",
                "patterns_good",
                "patterns_bad",
                "archetype_good",
                "archetype_bad",
                "featureDbScanClustersArchetypeEnabled",
                "featureDbScanClustersPatternsEnabled",
            ],
        ),
        (
            "shapes",
            ARCH_LR2_CLASSIFIER,
            1,
            [
                "good",
                "bad",
                "featureLengthRatio",
                "featureRegexAggregateDisabled",
                "archetype_good",
                "archetype_bad",
                "featureDbScanClustersArchetypeEnabled",
                "featureDbScanClustersPatternsEnabled",
            ],
        ),
        (
            "shapes",
            ARCH_LR2_CLASSIFIER,
            0,
            [
                "good",
                "bad",
                "featureLengthRatio",
                "featureRegexAggregateDisabled",
                "featureDbScanClustersArchetypeEnabled",
                "featureDbScanClustersPatternsEnabled",
            ],
        ),
    ],
)
def test_generates_features_when_env_train_quality_2(
    lesson: str,
    arch: str,
    train_quality: int,
    required_fields: List[str],
    tmpdir,
    data_root: str,
    shared_root: str,
    monkeypatch,
):
    monkeypatch.setenv("TRAIN_QUALITY_DEFAULT", str(train_quality))
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        _, model_name = path.split(train_result.models)

        cm = find_predicton_config_and_pickle(
            ModelRef(
                arch=ARCH_LR2_CLASSIFIER,
                lesson=model_name,
                filename=MODEL_FILE_NAME,
            ),
            test_config.find_data_dao(),
        )

        fields_in_config = frozenset(cm.config.get_expectation("0").features.keys())
        assert fields_in_config == frozenset(
            required_fields
        ), f"Config file does not contain exact features as required, Expected {list(required_fields)} got {list(fields_in_config)}"
