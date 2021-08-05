#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path

import pytest

from opentutor_classifier import AnswerClassifierInput
from opentutor_classifier.config import (
    LABEL_BAD,
    LABEL_GOOD,
    LABEL_NEUTRAL,
    LABEL_UNSPECIFIED,
    confidence_threshold_default,
)

from .utils import fixture_path, read_test_set_from_csv
from .types import (
    ComparisonType,
    _TestExample,
    _TestExpectation,
    _TestSet,
)

"""
"0",a neutral response for exp "0",neutral
"0",no label is also neutral in a test set,
1,a good response for exp 1,good
"""

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.mark.parametrize(
    "testset_name,confidence_threshold,expected",
    [
        (
            "testset_test",
            CONFIDENCE_THRESHOLD_DEFAULT,
            _TestSet(
                examples=[
                    _TestExample(
                        input=AnswerClassifierInput(
                            expectation="0", input_sentence="a good response for exp 0"
                        ),
                        expectations=[
                            _TestExpectation(
                                expectation="0",
                                evaluation=LABEL_GOOD,
                                score=CONFIDENCE_THRESHOLD_DEFAULT,
                                comparison=ComparisonType.GTE,
                            )
                        ],
                    ),
                    _TestExample(
                        input=AnswerClassifierInput(
                            expectation="0", input_sentence="a bad response for exp 0"
                        ),
                        expectations=[
                            _TestExpectation(
                                expectation="0",
                                evaluation=LABEL_BAD,
                                score=CONFIDENCE_THRESHOLD_DEFAULT,
                                comparison=ComparisonType.GTE,
                            )
                        ],
                    ),
                    _TestExample(
                        input=AnswerClassifierInput(
                            expectation="0",
                            input_sentence="a neutral response for exp 0",
                        ),
                        expectations=[
                            _TestExpectation(
                                expectation="0",
                                evaluation=LABEL_NEUTRAL,
                                score=CONFIDENCE_THRESHOLD_DEFAULT,
                                comparison=ComparisonType.LT,
                            )
                        ],
                    ),
                    _TestExample(
                        input=AnswerClassifierInput(
                            expectation="0",
                            input_sentence="no label is also neutral in a test set",
                        ),
                        expectations=[
                            _TestExpectation(
                                expectation="0",
                                evaluation=LABEL_UNSPECIFIED,
                                score=CONFIDENCE_THRESHOLD_DEFAULT,
                                comparison=ComparisonType.LT,
                            )
                        ],
                    ),
                    _TestExample(
                        input=AnswerClassifierInput(
                            expectation="1", input_sentence="a good response for exp 1"
                        ),
                        expectations=[
                            _TestExpectation(
                                expectation="1",
                                evaluation=LABEL_GOOD,
                                score=CONFIDENCE_THRESHOLD_DEFAULT,
                                comparison=ComparisonType.GTE,
                            )
                        ],
                    ),
                ]
            ),
        )
    ],
)
def test_parses_a_testset_from_csv(
    testset_name: str, confidence_threshold: float, expected: _TestSet
):
    observed = read_test_set_from_csv(
        fixture_path(path.join("data", testset_name, "test.csv")), confidence_threshold
    )
    assert expected.to_dict() == observed.to_dict()
