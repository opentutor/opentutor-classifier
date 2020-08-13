#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import pytest
from opentutor_classifier import (
    AnswerClassifierInput,
    ExpectationClassifierResult,
    SpeechActClassifierResult,
)
from opentutor_classifier.svm import SVMAnswerClassifier, load_config_into_objects
from . import fixture_path
import os


@pytest.fixture(scope="module")
def model_root() -> str:
    return fixture_path("models")


@pytest.fixture(scope="module")
def shared_root() -> str:
    return fixture_path("shared")


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,config_data,expected_results",
    [
        (
            "peer pressure can change your behavior",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.99, evaluation="Good")],
        )
    ],
)
def test_evaluates_one_expectation_for_q1(
    model_root,
    shared_root,
    input_answer,
    input_expectation_number,
    config_data,
    expected_results,
):
    model_root = os.path.join(model_root, "question1")
    classifier = SVMAnswerClassifier(model_root=model_root, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer,
            config_data=load_config_into_objects(config_data),
            expectation=input_expectation_number,
        )
    )
    assert len(result.expectation_results) == len(expected_results)
    for res, res_expected in zip(result.expectation_results, expected_results):
        assert res.expectation == res_expected.expectation
        assert round(res.score, 2) == res_expected.score
        assert res.evaluation == res_expected.evaluation


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,config_data,expected_results",
    [
        (
            "they need sunlight",
            0,
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
        ),
        (
            "peer pressure",
            0,
            {
                "question": "What are the challenges to demonstrating integrity in a group?",
                "expectations": [
                    {
                        "ideal": "Peer pressure can cause you to allow inappropriate behavior"
                    },
                    {"ideal": "Enforcing the rules can make you unpopular"},
                ],
            },
            [
                ExpectationClassifierResult(
                    expectation=0, evaluation="Bad", score=0.01
                ),
                ExpectationClassifierResult(
                    expectation=1, evaluation="Bad", score=0.17
                ),
            ],
        ),
        (
            "influence from others can change your behavior",
            0,
            {
                "question": "What are the challenges to demonstrating integrity in a group?",
                "expectations": [
                    {
                        "ideal": "Peer pressure can cause you to allow inappropriate behavior"
                    }
                ],
            },
            [ExpectationClassifierResult(expectation=0, evaluation="Bad", score=0.01)],
        ),
        (
            "hi",
            0,
            {
                "question": "What are the challenges to demonstrating integrity in a group?",
                "expectations": [
                    {
                        "ideal": "Peer pressure can cause you to allow inappropriate behavior"
                    }
                ],
            },
            [ExpectationClassifierResult(expectation=0, evaluation="Bad", score=0.14)],
        ),
        (
            "some gibberish kjlsdafhalkjfha",
            0,
            {
                "question": "What are the challenges to demonstrating integrity in a group?",
                "expectations": [
                    {
                        "ideal": "Peer pressure can cause you to allow inappropriate behavior"
                    }
                ],
            },
            [ExpectationClassifierResult(expectation=0, evaluation="Bad", score=0.14)],
        ),
    ],
)
def test_evaluates_for_default_model(
    model_root,
    shared_root,
    input_answer,
    input_expectation_number,
    config_data,
    expected_results,
):
    model_root = os.path.join(model_root, "default")
    classifier = SVMAnswerClassifier(model_root=model_root, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer,
            config_data=load_config_into_objects(config_data),
            expectation=input_expectation_number,
        )
    )
    assert len(result.expectation_results) == len(expected_results)
    for res, res_expected in zip(result.expectation_results, expected_results):
        assert res.expectation == res_expected.expectation
        assert round(res.score, 2) == res_expected.score
        assert res.evaluation == res_expected.evaluation


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,config_data,expected_results",
    [
        (
            "Current flows in the same direction as the arrow",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.96, evaluation="Good")],
        )
    ],
)
def test_evaluates_one_expectation_for_q2(
    model_root,
    shared_root,
    input_answer,
    input_expectation_number,
    config_data,
    expected_results,
):
    model_root = os.path.join(model_root, "question2")
    classifier = SVMAnswerClassifier(model_root=model_root, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer,
            config_data=load_config_into_objects(config_data),
            expectation=input_expectation_number,
        )
    )
    assert len(result.expectation_results) == len(expected_results)
    for res, res_expected in zip(result.expectation_results, expected_results):
        assert res.expectation == res_expected.expectation
        assert round(res.score, 2) == res_expected.score
        assert res.evaluation == res_expected.evaluation


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,config_data,expected_results",
    [
        (
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
        )
    ],
)
def test_evaluates_with_no_input_expectation_number_for_q1(
    model_root,
    shared_root,
    input_answer,
    input_expectation_number,
    config_data,
    expected_results,
):
    model_root = os.path.join(model_root, "question1")
    classifier = SVMAnswerClassifier(model_root=model_root, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer,
            config_data=load_config_into_objects(config_data),
            expectation=input_expectation_number,
        )
    )
    assert len(result.expectation_results) == len(expected_results)
    for res, res_expected in zip(result.expectation_results, expected_results):
        assert res.expectation == res_expected.expectation
        assert round(res.score, 2) == res_expected.score
        assert res.evaluation == res_expected.evaluation


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,config_data,expected_results,expected_sa_results",
    [
        (
            "I dont know what you are talking about",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.86, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "I do not understand",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.87, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "I believe the answer is peer pressure can change your behavior",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.99, evaluation="Good")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "Fuck you tutor",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.95, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "What the hell is that?",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.96, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "I dont know this shit",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.86, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "I dont know this shit but I guess the answer is peer pressure can change your behavior",
            0,
            {},
            [ExpectationClassifierResult(expectation=0, score=0.99, evaluation="Good")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
    ],
)
def test_evaluates_meta_cognitive_sentences(
    model_root,
    shared_root,
    input_answer,
    input_expectation_number,
    config_data,
    expected_results,
    expected_sa_results,
):
    model_root = os.path.join(model_root, "question1")
    classifier = SVMAnswerClassifier(model_root=model_root, shared_root=shared_root)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer,
            config_data=load_config_into_objects(config_data),
            expectation=input_expectation_number,
        )
    )
    assert len(result.expectation_results) == len(expected_results)
    assert (
        expected_sa_results["metacognitive"].evaluation
        == result.speech_acts["metacognitive"].evaluation
    )
    assert (
        expected_sa_results["metacognitive"].score
        == result.speech_acts["metacognitive"].score
    )
    assert (
        expected_sa_results["profanity"].evaluation
        == result.speech_acts["profanity"].evaluation
    )
    assert (
        expected_sa_results["profanity"].score == result.speech_acts["profanity"].score
    )

    for res, res_expected in zip(result.expectation_results, expected_results):
        assert res.expectation == res_expected.expectation
        assert round(res.score, 2) == res_expected.score
        assert res.evaluation == res_expected.evaluation
