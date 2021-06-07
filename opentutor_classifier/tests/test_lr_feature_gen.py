#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path

import pytest

from opentutor_classifier.lr.expectations import preprocess_sentence
from opentutor_classifier.lr.clustering_features import CustomAgglomerativeClustering

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
    patterns = CustomAgglomerativeClustering.deduplicate_patterns(
        input_patterns_with_fpr, cuttoff_fpr
    )
    assert patterns == expected_patterns, f"Expected {expected_patterns} got {patterns}"
