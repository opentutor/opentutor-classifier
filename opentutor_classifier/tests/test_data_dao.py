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

from opentutor_classifier import (
    ARCH_LR_CLASSIFIER,
    ExpectationConfig,
    QuestionConfig,
    QuestionConfigSaveReq,
)
from opentutor_classifier.api import get_graphql_endpoint, update_features_gql
import opentutor_classifier.dao
from opentutor_classifier.utils import load_config


@pytest.mark.parametrize("arch", [(ARCH_LR_CLASSIFIER)])
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
