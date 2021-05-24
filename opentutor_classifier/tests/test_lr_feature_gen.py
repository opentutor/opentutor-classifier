#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path

import pytest

from opentutor_classifier import ARCH_LR_CLASSIFIER
from opentutor_classifier.utils import load_config
from opentutor_classifier.lr.expectations import preprocess_sentence

from typing import Dict, List, Any
from .utils import fixture_path, test_env_isolated, train_classifier


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
    ],
)
def test_text2num(sentence: str, expected_transformation: str):
    transformed_tranform = preprocess_sentence(sentence)
    assert (
        expected_transformation == transformed_tranform
    ), f"Expected {expected_transformation} got {transformed_tranform}"


@pytest.mark.parametrize(
    "lesson,expected_features",
    [
        # (
        #     "candles",
        #     [
        #         {
        #             "bad": [
        #                 "different|differ|unequal|(\\b(isn't|not)\\b.*(same|equal))"
        #             ],
        #             "good": ["constant|same|identical|equal|uniform"],
        #             "patterns_bad": [
        #                 "[NEG] + lengths",
        #                 "[NEG] + lit",
        #                 "burns + change",
        #                 "burns + covary",
        #                 "burns + one",
        #                 "burns + time",
        #                 "burns + well",
        #                 "change + covary",
        #                 "change + one",
        #                 "change + same",
        #                 "change + time",
        #                 "change + well",
        #                 "covary",
        #                 "invariant + lengths",
        #                 "invariant + lit",
        #                 "invariant + same",
        #                 "invariant + time",
        #                 "lengths",
        #                 "lit",
        #                 "one",
        #                 "same + well",
        #                 "time + well",
        #                 "well",
        #             ],
        #             "patterns_good": [
        #                 "constant + indicates",
        #                 "constant + rate",
        #                 "constant + uniformity",
        #                 "indicates",
        #                 "pace",
        #                 "rate + uniformity",
        #                 "same + uniform",
        #                 "uniformity",
        #             ],
        #         },
        #         {
        #             "bad": [
        #                 "\\b(not|isn't|no|without)\\b.*\\b(related|relationship|change)\\b"
        #             ],
        #             "good": [
        #                 "(covary|covariance|co-variance|co-vary)|\\b(same|equal|together)\\b.*\\b(change|increase|up|grow|vary|rate)\\b|(increase|change|grow|up|vary|rate).*(same|together|equal)"
        #             ],
        #             "patterns_bad": [
        #                 "[NEG] + lengths",
        #                 "[NEG] + lit",
        #                 "invariant + lengths",
        #                 "invariant + lit",
        #                 "invariant + same",
        #                 "invariant + time",
        #                 "lengths + lit",
        #                 "lengths + proportional",
        #                 "lengths + ratio",
        #                 "lengths + relationship",
        #                 "lengths + same",
        #                 "lengths + time",
        #                 "lit",
        #             ],
        #             "patterns_good": ["also", "covariates", "increases", "length"],
        #         },
        #         {
        #             "bad": ["multiples"],
        #             "good": [
        #                 "(\\b(not|isn't|no|without)\\b.*(fixed|constant|same).*(ratio|rate|multiple))|((ratio|rate|multiple).*\\b(not|isn't|no|without)\\b.*(fixed|constant|same))"
        #             ],
        #             "patterns_bad": [
        #                 "[NEG] + covary",
        #                 "candles + covary",
        #                 "candles + invariant",
        #                 "candles + uniform",
        #                 "covary",
        #                 "however + invariant",
        #                 "however + uniform",
        #                 "invariant + uniform",
        #             ],
        #             "patterns_good": [],
        #         },
        #         {
        #             "bad": ["proportion"],
        #             "good": ["\\b(no|not|isn't|hasn't|never|without)\\b.*proportion"],
        #             "patterns_bad": [
        #                 "[NEG] + ratios",
        #                 "both + burning",
        #                 "both + covary",
        #                 "burn + invariant",
        #                 "burn + ratios",
        #                 "burning + covary",
        #                 "burning + keep",
        #                 "covary + keep",
        #                 "different + invariant",
        #                 "different + ratios",
        #                 "different + time",
        #                 "invariant + ratios",
        #                 "rate + ratios",
        #                 "ratios",
        #             ],
        #             "patterns_good": ["[NEG] + proportional"],
        #         },
        #     ],
        # ),
        # (
        #     "ies-rectangle",
        #     [
        #         {
        #             "bad": ["less|different|any"],
        #             "good": ["1|more"],
        #             "patterns_bad": [
        #                 "3 + impact",
        #                 "37 + impact",
        #                 "40 + impact",
        #                 "bigger + impact",
        #                 "bigger + unit",
        #                 "bigger + x",
        #                 "closer + impact",
        #                 "closer + x",
        #                 "difference + impact",
        #                 "difference + x",
        #                 "gets + impact",
        #                 "gets + unit",
        #                 "gets + x",
        #                 "impact",
        #                 "rectangle + x",
        #                 "square + x",
        #             ],
        #             "patterns_good": [
        #                 "1 + closer",
        #                 "1 + gets",
        #                 "1 + like",
        #                 "1 + looks",
        #                 "1 + ratio",
        #                 "1 + rectangle",
        #                 "1 + square",
        #                 "closer + gets",
        #                 "closer + like",
        #                 "closer + looks",
        #                 "closer + ratio",
        #                 "closer + rectangle",
        #                 "closer + square",
        #                 "gets + like",
        #                 "gets + looks",
        #                 "gets + ratio",
        #                 "gets + rectangle",
        #                 "gets + square",
        #                 "like + ratio",
        #                 "like + rectangle",
        #                 "looks + ratio",
        #                 "looks + rectangle",
        #                 "ratio + rectangle",
        #                 "ratio + square",
        #                 "rectangle + square",
        #             ],
        #         },
        #         {
        #             "bad": ["same"],
        #             "good": ["less"],
        #             "patterns_bad": [],
        #             "patterns_good": [
        #                 "3",
        #                 "3 + bigger",
        #                 "3 + difference",
        #                 "3 + effect",
        #                 "3 + less",
        #                 "3 + rectangle",
        #                 "3 + unit",
        #                 "bigger",
        #                 "bigger + difference",
        #                 "bigger + effect",
        #                 "bigger + less",
        #                 "bigger + rectangle",
        #                 "bigger + unit",
        #                 "difference",
        #                 "difference + effect",
        #                 "difference + less",
        #                 "difference + rectangle",
        #                 "difference + unit",
        #                 "effect",
        #                 "effect + less",
        #                 "effect + rectangle",
        #                 "effect + unit",
        #                 "less + rectangle",
        #                 "less + unit",
        #                 "rectangle + unit",
        #                 "unit",
        #             ],
        #         },
        #         {
        #             "bad": ["10|17|20", "27|30"],
        #             "good": ["37|40"],
        #             "patterns_bad": [],
        #             "patterns_good": ["37"],
        #         },
        #     ],
        # ),
    ],
)
@pytest.mark.slow
def test_generates_features(
    lesson: str,
    expected_features: List[Dict[str, Any]],
    tmpdir,
    data_root: str,
    shared_root: str,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=ARCH_LR_CLASSIFIER, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)
        config_file = path.join(
            test_config.output_dir, test_config.arch, lesson, "config.yaml"
        )
        assert path.isfile(config_file)
        generated_config = load_config(config_file)
        # "'s" check
        print(generated_config.expectations)
        for e in generated_config.expectations:
            for pattern in e.features["patterns_good"]:
                assert "'s'" not in pattern, f"'s found in {pattern}"
            for pattern in e.features["patterns_bad"]:
                assert "'s" not in pattern, f"'s found in {pattern}"
        assert [e.features for e in generated_config.expectations] == expected_features
