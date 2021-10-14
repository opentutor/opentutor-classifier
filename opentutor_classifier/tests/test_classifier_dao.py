#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import responses
from opentutor_classifier.classifier_dao import ClassifierDao
from opentutor_classifier import (
    ARCH_LR2_CLASSIFIER,
    ClassifierConfig,
)
from os import path

import pytest
from tests.utils import (
    example_data_path,
    fixture_path,
    mocked_data_dao,
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
    "lesson,arch",
    [
        ("question1", ARCH_LR2_CLASSIFIER),
    ],
)
@responses.activate
def test_classifier_cache(arch: str, lesson: str, tmpdir, data_root, shared_root):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)
        mocked_data_dao(
            lesson,
            example_data_path(""),
            test_config.output_dir,
            test_config.output_dir,
        )
        config = ClassifierConfig(
            dao=test_config.find_data_dao(),
            model_name=lesson,
            shared_root=shared_root,
            model_roots=[test_config.output_dir],
        )

        dao = ClassifierDao()
        classifier1 = dao.find_classifier(lesson, config, arch)
        classifier2 = dao.find_classifier(lesson, config, arch)
        assert classifier1 == classifier2

        train_classifier(lesson, test_config)

        classifier3 = dao.find_classifier(lesson, config, arch)
        assert classifier3 != classifier1
