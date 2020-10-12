#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json

import pytest

from opentutor_classifier import ExpectationClassifierResult, SpeechActClassifierResult
from . import fixture_path


@pytest.fixture(autouse=True)
def python_path_env(monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", fixture_path("models"))
    monkeypatch.setenv("SHARED_ROOT", fixture_path("shared"))


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


def test_returns_404_response_with_no_question_available_with_no_config_data(client):
    res = client.post(
        "/classifier/evaluate/",
        data=json.dumps({"lesson": "doesNotExist", "input": "peer pressure"}),
        content_type="application/json",
    )
    assert res.status_code == 404
    assert res.json == {
        "message": "No models found for lesson doesNotExist. Config data is required"
    }


@pytest.mark.parametrize(
    "input_lesson, input_answer, input_expectation, config_data, expected_results",
    [
        # (
        #     "doesNotExist",
        #     "peer pressure can change your behavior",
        #     0,
        #     {
        #         "question": "What are the challenges to demonstrating integrity in a group?",
        #         "expectations": [
        #             {
        #                 "ideal": "Peer pressure can cause you to allow inappropriate behavior"
        #             }
        #         ],
        #     },
        #     [ExpectationClassifierResult(expectation=0, score=0.0, evaluation="Bad")],
        # ),
        (
            "doesNotExist",
            "they need sunlight",
            -1,
            {
                "question": "how can i grow better plants?",
                "expectations": [
                    {"ideal": "give them the right amount of water"},
                    {"ideal": "they need sunlight"},
                ],
            },
            [
                ExpectationClassifierResult(
                    expectation=0, evaluation="Bad", score=0.02
                ),
                ExpectationClassifierResult(
                    expectation=1, evaluation="Good", score=1.0
                ),
            ],
        )
    ],
)
def test_evaluate_with_no_question_available_with_config_data(
    client, input_lesson, input_answer, input_expectation, config_data, expected_results
):
    res = client.post(
        "/classifier/evaluate/",
        data=json.dumps(
            {
                "lesson": input_lesson,
                "input": input_answer,
                "expectation": input_expectation,
                "config": config_data,
            }
        ),
        content_type="application/json",
    )
    assert res.status_code == 200
    assert res.json["version"]["modelId"] == "default"
    results = res.json["output"]["expectationResults"]
    assert len(results) == len(expected_results)
    for res, res_expected in zip(results, expected_results):
        assert res["expectation"] == res_expected.expectation
        assert round(float(res["score"]), 2) == res_expected.score
        assert res["evaluation"] == res_expected.evaluation


@pytest.mark.parametrize(
    "input_lesson,input_answer,input_expectation,config_data,expected_results,expected_sa_results",
    [
        (
            "q1",
            "peer pressure can change your behavior",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.99, evaluation="Good")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "q1",
            "peer pressure can change your behavior",
            -1,
            {},
            [
                ExpectationClassifierResult(
                    expectation=0, score=0.99, evaluation="Good"
                ),
                ExpectationClassifierResult(
                    expectation=1, score=0.50, evaluation="Bad"
                ),
                ExpectationClassifierResult(
                    expectation=2, score=0.57, evaluation="Bad"
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
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.86, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
    ],
)
def test_evaluate_classifies_user_answers(
    client,
    input_lesson,
    input_answer,
    input_expectation,
    config_data,
    expected_results,
    expected_sa_results,
):
    res = client.post(
        "/classifier/evaluate/",
        data=json.dumps(
            {
                "lesson": input_lesson,
                "input": input_answer,
                "expectation": input_expectation,
                "config": config_data,
            }
        ),
        content_type="application/json",
    )
    assert res.json["version"]["modelId"] == input_lesson
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
    assert speech_acts["profanity"]["score"] == expected_sa_results["profanity"].score
    results = res.json["output"]["expectationResults"]
    assert len(results) == len(expected_results)
    for res, res_expected in zip(results, expected_results):
        assert res["expectation"] == res_expected.expectation
        assert round(float(res["score"]), 2) == res_expected.score
        assert res["evaluation"] == res_expected.evaluation
