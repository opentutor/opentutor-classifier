#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
from os import path

import pytest
import responses

from typing import Dict, Any
from opentutor_classifier import (
    ARCH_LR2_CLASSIFIER,
    ArchLesson,
    ExpectationConfig,
    ModelRef,
    QuestionConfig,
    QuestionConfigSaveReq,
)
from opentutor_classifier.api import get_graphql_endpoint, update_features_gql
import opentutor_classifier.dao
from opentutor_classifier.lr2.constants import MODEL_FILE_NAME
from opentutor_classifier.utils import load_config
from tests.utils import fixture_path, test_env_isolated, train_classifier


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


@pytest.mark.parametrize(
    "input_dictionary,expected_output",
    [
        (
            {"GOOD": ["on", "off", "true", "false", "|"]},
            {"GOOD": ["|on", "|off", "|true", "|false", "||"]},
        )
    ],
)
def test_feature_escaping(
    input_dictionary: Dict[str, Any], expected_output: Dict[str, Any]
):
    expectation_config = ExpectationConfig("expectation_id", "ideal", input_dictionary)
    dictionary = expectation_config.to_dict()

    assert dictionary["features"] == expected_output
    assert expectation_config.features == input_dictionary

    copy_of_expectation_config = ExpectationConfig(**dictionary)
    assert copy_of_expectation_config.features == input_dictionary


@pytest.mark.parametrize("arch", [(ARCH_LR2_CLASSIFIER)])
@responses.activate
def test_saves_config_to_model_root_and_features_to_gql(arch: str, tmpdir, monkeypatch):
    model_root = tmpdir.mkdir("models")
    monkeypatch.setenv("MODEL_ROOT", model_root)
    responses.add(
        responses.POST,
        get_graphql_endpoint(),
        json={},
        status=200,
    )
    req = QuestionConfigSaveReq(
        arch=arch,
        lesson="lesson1",
        config=QuestionConfig(
            question="what type of thing is an apple?",
            expectations=[
                ExpectationConfig(
                    ideal="it is a fruit", features=dict(good=["a", "b"], bad="c")
                )
            ],
        ),
    )
    opentutor_classifier.dao.find_data_dao().save_config(req)
    assert json.loads(responses.calls[0].request.body) == update_features_gql(req)
    config_file = path.join(model_root, arch, req.lesson, "config.yaml")
    assert path.isfile(config_file)
    assert load_config(config_file).to_dict() == req.config.to_dict()


@pytest.mark.parametrize("lesson,arch", [("question1", ARCH_LR2_CLASSIFIER)])
def test_delete_model(tmpdir, data_root: str, shared_root: str, lesson: str, arch: str):
    with test_env_isolated(
        tmpdir, data_root, shared_root, lesson=lesson, arch=arch
    ) as test_config:
        result = train_classifier(lesson, test_config)
        assert path.exists(path.join(result.models, MODEL_FILE_NAME))
        assert path.exists(path.join(result.models, "config.yaml"))

    dao = test_config.find_data_dao()
    assert dao.trained_model_exists(ModelRef(arch, lesson, MODEL_FILE_NAME))
    dao.remove_trained_model(ArchLesson(arch=arch, lesson=lesson))
    assert not dao.trained_model_exists(
        ModelRef(arch=arch, lesson=lesson, filename=MODEL_FILE_NAME)
    )
