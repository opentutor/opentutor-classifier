#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
import json
from unittest.mock import patch, Mock

import pytest

from . import Bunch


@pytest.mark.parametrize(
    "classifier_domain,input_lesson,fake_task_id",
    [
        ("https://opentutor.org", "lesson_1", "fake_task_id_1"),
        ("http://a.diff.org", "lesson_2", "fake_task_id_2"),
    ],
)
@patch("opentutor_classifier_tasks.tasks.train_task")
def test_train(mock_train_task, classifier_domain, input_lesson, fake_task_id, client):
    mock_task = Bunch(id=fake_task_id)
    mock_train_task.apply_async.return_value = mock_task
    res = client.post(
        f"{classifier_domain}/classifier/train/",
        data=json.dumps({"lesson": input_lesson}),
        content_type="application/json",
    )
    assert res.status_code == 200
    assert res.json == {
        "data": {
            "id": fake_task_id,
            "lesson": input_lesson,
            "statusUrl": f"{classifier_domain}/classifier/train/status/{fake_task_id}",
        }
    }


# ISSUE: if the classifier api doesn't do end-to-end ssl
# (e.g. if nginx terminates ssl),
# then classifier-api doesn't know that its TRUE
# root url is https://...
@pytest.mark.parametrize(
    "request_root,env_val,expected_status_url_root",
    [
        ("http://opentutor.org", None, "http://opentutor.org"),
        ("http://opentutor.org", "1", "https://opentutor.org"),
        ("http://opentutor.org", "y", "https://opentutor.org"),
        ("http://opentutor.org", "true", "https://opentutor.org"),
        ("http://opentutor.org", "on", "https://opentutor.org"),
    ],
)
@patch("opentutor_classifier_tasks.tasks.train_task")
def test_env_fixes_ssl_status_url(
    mock_train_task: Mock,
    request_root: str,
    env_val: str,
    expected_status_url_root: str,
    monkeypatch,
    client,
):
    fake_task_id = "fake_task_id"
    fake_lesson_id = "lesson1"
    if env_val is not None:
        monkeypatch.setenv("STATUS_URL_FORCE_HTTPS", env_val)
    mock_task = Bunch(id=fake_task_id)
    mock_train_task.apply_async.return_value = mock_task
    res = client.post(
        f"{request_root}/classifier/train/",
        data=json.dumps({"lesson": fake_lesson_id}),
        content_type="application/json",
    )
    assert res.status_code == 200
    assert res.json == {
        "data": {
            "id": fake_task_id,
            "lesson": fake_lesson_id,
            "statusUrl": f"{expected_status_url_root}/classifier/train/status/fake_task_id",
        }
    }


@pytest.mark.parametrize(
    "task_id,state,status,info,expected_info",
    [
        ("fake-task-id-123", "PENDING", "working", None, None),
        ("fake-task-id-234", "STARTED", "working harder", None, None),
        (
            "fake-task-id-456",
            "SUCCESS",
            "done!",
            {"expectations": [{"accuracy": 0.81}, {"accuracy": 0.92}]},
            {"expectations": [{"accuracy": 0.81}, {"accuracy": 0.92}]},
        ),
        (
            "fake-task-id-678",
            "FAILURE",
            "error!",
            Exception("error message"),
            "error message",
        ),
    ],
)
@patch("opentutor_classifier_tasks.tasks.train_task")
def test_it_returns_status_for_a_train_job(
    mock_train_task, task_id, state, status, info, expected_info, client
):
    mock_task = Bunch(id=task_id, state=state, status=status, info=info)
    mock_train_task.AsyncResult.return_value = mock_task
    res = client.get(f"/classifier/train/status/{task_id}")
    assert res.status_code == 200
    assert res.json == {
        "data": {"id": task_id, "state": state, "status": status, "info": expected_info}
    }
