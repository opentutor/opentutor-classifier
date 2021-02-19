#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import pytest

from opentutor_classifier import (
    ARCH_SVM_CLASSIFIER,
    ARCH_LR_CLASSIFIER,
    ClassifierFactory,
    ClassifierConfig,
)
from opentutor_classifier.lr import LRAnswerClassifier
from opentutor_classifier.svm import SVMAnswerClassifier


@pytest.mark.parametrize(
    "arch,expected_classifier_type",
    [
        (ARCH_SVM_CLASSIFIER, SVMAnswerClassifier),
        (ARCH_LR_CLASSIFIER, LRAnswerClassifier),
    ],
)
def test_creates_a_classifier_with_default_arch(
    monkeypatch, arch: str, expected_classifier_type
):
    monkeypatch.setenv("CLASSIFIER_ARCH", arch)
    assert isinstance(
        ClassifierFactory().new_classifier(ClassifierConfig(model_name="somemodel")),
        expected_classifier_type,
    )
