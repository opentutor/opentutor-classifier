#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from contextlib import contextmanager
import pytest
import responses
import json
from . import fixture_path
import os


@contextmanager
def _mock_healthchecks(
    gql_fixture_name: str,
    gql_url="http://graphql/graphql",
    gql_status=200,
    admin_url="http://admin",
    admin_status=200,
    home_url="http://home",
    home_status=200,
    tutor_url="http://tutor",
    tutor_status=200,
):
    rsps = responses.RequestsMock()
    try:
        rsps.start()
        with open(
            fixture_path(os.path.join("graphql", f"{gql_fixture_name}.json"))
        ) as f:
            data = json.load(f)
            rsps.add(responses.POST, gql_url, json=data, status=gql_status)
        rsps.add(responses.HEAD, admin_url, status=admin_status)
        rsps.add(responses.HEAD, home_url, status=home_status)
        rsps.add(responses.HEAD, tutor_url, status=tutor_status)
        # responses.add(responses.GET, "http://training/ping", status=200)
        yield None
    finally:
        rsps.stop()


@pytest.mark.parametrize(
    "message,status",
    [("pong!", "success")],
)
def test_healthcheck_returns_all_statuses(client, message, status):
    with _mock_healthchecks("admin_ok"):
        res = client.get("/classifier/healthcheck/")
        assert len(res.json.get("services")) == 5


@pytest.mark.parametrize(
    "message,status",
    [("pong!", "success")],
)
def test_200_if_all_healthy(client, message, status):
    with _mock_healthchecks("admin_ok"):
        res = client.get("/classifier/healthcheck/")
        assert res.json["services"]["graphql"]["status"] == 200
        assert res.json["services"]["admin"]["status"] == 200
        assert res.json["services"]["home"]["status"] == 200
        assert res.json["services"]["tutor"]["status"] == 200
        # assert res.json["services"]["training"]["status"] == 200
        assert res.status_code == 200


@pytest.mark.parametrize(
    "message,status",
    [("pong!", "success")],
)
def test_503_if_not_healthy(client, message, status):
    with _mock_healthchecks(
        "admin_bad", gql_status=404, admin_status=400, home_status=500, tutor_status=502
    ):
        res = client.get("/classifier/healthcheck/")
        assert res.json["services"]["graphql"]["status"] == 404
        assert res.json["services"]["admin"]["status"] == 400
        assert res.json["services"]["home"]["status"] == 500
        assert res.json["services"]["tutor"]["status"] == 502
        # assert res.json["services"]["training"]["status"] == 403
        assert res.status_code == 503


@pytest.mark.parametrize(
    "message,status",
    [("pong!", "success")],
)
def test_can_override_healthcheck_admin_url(
    client, message, status, monkeypatch
) -> None:
    with _mock_healthchecks("admin_ok", admin_url="http://someadmin", admin_status=418):
        monkeypatch.setenv("HEALTHCHECK_ADMIN", "http://someadmin")
        res = client.get("/classifier/healthcheck/")
        assert res.json["services"]["admin"]["status"] == 418
        assert res.status_code == 503


@pytest.mark.parametrize(
    "message,status",
    [("pong!", "success")],
)
def test_can_override_healthcheck_home_url(
    client, message, status, monkeypatch
) -> None:
    with _mock_healthchecks("admin_ok", home_url="http://somehome", home_status=418):
        monkeypatch.setenv("HEALTHCHECK_HOME", "http://somehome")
        res = client.get("/classifier/healthcheck/")
        assert res.json["services"]["home"]["status"] == 418
        assert res.status_code == 503


@pytest.mark.parametrize(
    "message,status",
    [("pong!", "success")],
)
def test_can_override_healthcheck_tutor_url(
    client, message, status, monkeypatch
) -> None:
    with _mock_healthchecks("admin_ok", tutor_url="http://sometutor", tutor_status=418):
        monkeypatch.setenv("HEALTHCHECK_TUTOR", "http://sometutor")
        res = client.get("/classifier/healthcheck/")
        assert res.json["services"]["tutor"]["status"] == 418
        assert res.status_code == 503
