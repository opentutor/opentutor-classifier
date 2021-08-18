#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import os
from typing import List

import pytest

from opentutor_classifier import ExpectationClassifierResult, SpeechActClassifierResult  # type: ignore
import responses  # type: ignore
from . import fixture_path
from .utils import mocked_data_dao


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return os.path.dirname(word2vec)


@pytest.fixture(autouse=True)
def python_path_env(monkeypatch, shared_root):
    monkeypatch.setenv("MODEL_ROOT", fixture_path("models"))
    monkeypatch.setenv("MODEL_DEPLOYED_ROOT", fixture_path("models_deployed"))
    monkeypatch.setenv("SHARED_ROOT", shared_root)


def test_returns_400_response_when_lesson_not_set(client):
    res = client.post(
        "/classifier/evaluate/",
        data=json.dumps({"input": "peer pressure", "expectation": "0"}),
        content_type="application/json",
    )
    assert res.status_code == 400
    assert res.json == {"lesson": ["required field"]}


def test_returns_400_response_when_input_not_set(client):
    res = client.post(
        "/classifier/evaluate/",
        data=json.dumps({"lesson": "q1", "expectation": "0"}),
        content_type="application/json",
    )
    assert res.status_code == 400
    assert res.json == {"input": ["required field"]}


@pytest.mark.parametrize(
    "lesson, answer, expectation, expected_results",
    [
        (
            "q1-untrained",
            "peer pressure might lead to bad behavior",
            "0",
            [
                ExpectationClassifierResult(
                    expectation_id="0", evaluation="Good", score=0.62
                ),
            ],
        )
    ],
)
@responses.activate
def test_evaluate_uses_default_model_when_question_untrained(
    client,
    lesson: str,
    answer: str,
    expectation: str,
    expected_results: List[ExpectationClassifierResult],
):
    with mocked_data_dao(
        lesson,
        fixture_path("data"),
        fixture_path("models"),
        fixture_path("models_deployed"),
    ):
        res = client.post(
            "/classifier/evaluate/",
            data=json.dumps(
                {
                    "lesson": lesson,
                    "input": answer,
                    "expectation": expectation,
                    "config": {},
                }
            ),
            content_type="application/json",
        )
        assert res.status_code == 200
        results = res.json["output"]["expectationResults"]
        assert len(results) == len(expected_results)
        for res, res_expected in zip(results, expected_results):
            assert res["expectationId"] == res_expected.expectation_id
            assert round(float(res["score"]), 2) == res_expected.score
            assert res["evaluation"] == res_expected.evaluation


@pytest.mark.parametrize(
    "lesson,answer,expectation,config_data,expected_results,expected_sa_results",
    [
        (
            "q1",
            "peer pressure can change your behavior",
            "0",
            {},
            [
                ExpectationClassifierResult(
                    expectation_id="0", score=0.76, evaluation="Good"
                )
            ],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "q1",
            "peer pressure can change your behavior",
            "",
            {},
            [
                ExpectationClassifierResult(
                    expectation_id="0", score=0.76, evaluation="Good"
                ),
                ExpectationClassifierResult(
                    expectation_id="1", score=0.56, evaluation="Good"
                ),
                ExpectationClassifierResult(
                    expectation_id="2", score=0.6, evaluation="Bad"
                ),
            ],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "q1",
            "I dont know what you are talking about",
            "0",
            {},
            [
                ExpectationClassifierResult(
                    expectation_id="0", score=0.52, evaluation="Bad"
                )
            ],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
    ],
)
@responses.activate
def test_evaluate_classifies_user_answers(
    client,
    lesson,
    answer,
    expectation,
    config_data,
    expected_results,
    expected_sa_results,
):
    with mocked_data_dao(
        lesson,
        fixture_path("data"),
        fixture_path("models"),
        fixture_path("models_deployed"),
    ):
        res = client.post(
            "/classifier/evaluate/",
            data=json.dumps(
                {
                    "lesson": lesson,
                    "input": answer,
                    "expectation": expectation,
                    "config": config_data,
                }
            ),
            content_type="application/json",
        )
        speech_acts = res.json["output"]["speechActs"]
        assert (
            speech_acts["metacognitive"]["evaluation"]
            == expected_sa_results["metacognitive"].evaluation
        )
        assert (
            speech_acts["metacognitive"]["score"]
            == expected_sa_results["metacognitive"].score
        )
        assert (
            speech_acts["profanity"]["evaluation"]
            == expected_sa_results["profanity"].evaluation
        )
        assert (
            speech_acts["profanity"]["score"] == expected_sa_results["profanity"].score
        )
        results = res.json["output"]["expectationResults"]
        assert len(results) == len(expected_results)
        for res, res_expected in zip(results, expected_results):
            assert res["expectationId"] == res_expected.expectation_id
            assert round(float(res["score"]), 2) == res_expected.score
            assert res["evaluation"] == res_expected.evaluation
