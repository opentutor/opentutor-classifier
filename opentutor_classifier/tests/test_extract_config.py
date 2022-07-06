#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
from typing import List

import pytest

from opentutor_classifier import (
    ClassifierConfig,
    ClassifierFactory,
)
from opentutor_classifier.config import confidence_threshold_default
from .utils import (
    fixture_path,
    test_env_isolated,
    train_classifier,
)

import opentutor_classifier.dao

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


def _test_train_and_check_config(
    lesson: str,
    arch: str,
    fields: List[str],
    tmpdir,
    data_root: str,
    shared_root: str,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)
        model_root, model_name = path.split(train_result.models)
        classifier = ClassifierFactory().new_classifier(
            ClassifierConfig(
                dao=opentutor_classifier.dao.find_data_dao(),
                model_name=model_name,
                model_roots=[model_root],
                shared_root=shared_root,
            ),
        )

        config = classifier.save_config_and_model()
        for field in fields:
            assert field in config
