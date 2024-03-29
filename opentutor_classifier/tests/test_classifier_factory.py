#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import pytest

from opentutor_classifier import (
    ARCH_LR2_CLASSIFIER,
    ClassifierFactory,
    ClassifierConfig,
)
import opentutor_classifier.dao
from opentutor_classifier.lr2 import LRAnswerClassifier


@pytest.mark.parametrize(
    "arch,expected_classifier_type",
    [
        (ARCH_LR2_CLASSIFIER, LRAnswerClassifier),
    ],
)
def test_creates_a_classifier_with_default_arch(
    monkeypatch, arch: str, expected_classifier_type
):
    monkeypatch.setenv("CLASSIFIER_ARCH", arch)
    assert isinstance(
        ClassifierFactory().new_classifier(
            ClassifierConfig(
                dao=opentutor_classifier.dao.find_data_dao(), model_name="somemodel"
            )
        ),
        expected_classifier_type,
    )
