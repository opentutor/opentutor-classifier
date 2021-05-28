#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import pytest
import responses
import json
from . import fixture_path
from opentutor_classifier.api import get_graphql_endpoint

# def test_healthcheck_returns_all_statuses(client):
#     res = client.get("/classifier/healthcheck/")
#     assert len(res.json.get("services")) == 5


# def test_healthcheck_returns_200_responses_if_healthy(client):
#     res = client.get("/classifier/healthcheck/")
#     assert res.status_code == 200
#     assert res.json.get("services").get("admin").get("status") == 200
#     assert res.json.get("services").get("graphql").get("status") == 200
#     assert res.json.get("services").get("home").get("status") == 200
#     assert res.json.get("services").get("tutor").get("status") == 200
#     assert res.json.get("services").get("training").get("status") == 200


# def test_healthcheck_returns_503_if_unhealthy(client):
#     res = client.get("/classifier/healthcheck/")
#     assert res.status_code == 503
#     assert (
#         res.json.get("services").get("admin").get("status") != 200
#         or res.json.get("services").get("graphql").get("status") != 200
#         or res.json.get("services").get("home").get("status") != 200
#         or res.json.get("services").get("tutor").get("status") != 200
#         or res.json.get("services").get("training").get("status") != 200
#     )


@responses.activate
@pytest.mark.parametrize(
    "message,status",
    [
        ('pong!','success')
    ],
)
def test_can_i_mock(
    client, message, status
):
    with open(fixture_path("graphql/admin_ok.json")) as f:
        data = json.load(f)
    responses.add(responses.POST, get_graphql_endpoint(), json=data, status=200)
    res = client.get(f"/classifier/healthcheck/")
    assert res.json.get("services").get("graphql").get("status") == 200
