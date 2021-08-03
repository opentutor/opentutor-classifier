#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path

import pytest

from opentutor_classifier.lr.features import preprocess_sentence
from opentutor_classifier.lr.clustering_features import CustomDBScanClustering

from typing import List, Tuple
from .utils import fixture_path


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
def test_text2num(sentence: str, expected_transformation: str):
    transformed_tranform = preprocess_sentence(sentence)
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
            ["uniform"],
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
            ["37 + 40", "37", "sides + closer", "1 + ratio"],
        )
    ],
)
def test_univariate_selection(
    patterns: List[str],
    input_x: List[str],
    input_y: List[str],
    n: int,
    expected_patterns: List[str],
):
    patterns = CustomDBScanClustering.univariate_feature_selection(
        patterns, input_x, input_y, n
    )
    assert patterns == expected_patterns, f"Expected {expected_patterns} got {patterns}"
