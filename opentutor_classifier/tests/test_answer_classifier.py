#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
from typing import List

import pytest
import responses

from opentutor_classifier import (
    ARCH_SVM_CLASSIFIER,
    AnswerClassifierInput,
    ClassifierConfig,
    ClassifierFactory,
    SpeechActClassifierResult,
)
from opentutor_classifier.config import confidence_threshold_default
import opentutor_classifier.dao
from opentutor_classifier.utils import dict_to_config
from .utils import (
    assert_classifier_evaluate,
    assert_testset_accuracy,
    fixture_path,
    example_data_path,
    mocked_data_dao,
    read_example_testset,
)
from .types import ComparisonType, _TestExpectation


CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.fixture(scope="module")
def model_roots() -> List[str]:
    return [fixture_path("models"), fixture_path("models_deployed")]


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return os.path.dirname(word2vec)


@pytest.mark.parametrize(
    "lesson,arch,confidence_threshold,expected_accuracy",
    [
        ("question1", ARCH_SVM_CLASSIFIER, CONFIDENCE_THRESHOLD_DEFAULT, 1.0),
        ("question2", ARCH_SVM_CLASSIFIER, CONFIDENCE_THRESHOLD_DEFAULT, 1.0),
    ],
)
@responses.activate
def test_evaluate_example(
    model_roots,
    shared_root,
    lesson: str,
    arch: str,
    confidence_threshold: float,
    expected_accuracy: float,
):
    testset = read_example_testset(lesson, confidence_threshold=confidence_threshold)
    with mocked_data_dao(lesson, example_data_path(""), model_roots[0], model_roots[1]):
        assert_testset_accuracy(
            arch,
            os.path.join(model_roots[0], lesson),
            shared_root,
            testset,
            expected_accuracy,
        )


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,config_data,expected_results",
    [
        (
            "peer pressure leads you to allow bad behavior",
            -1,
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
                _TestExpectation(expectation=0, evaluation="Good", score=0.8),
                # # NOTE: this exp is incorrectly getting GOOD with very high confidence
                # _TestExpectation(
                #     expectation=1,
                #     score=CONFIDENCE_THRESHOLD_DEFAULT,
                #     comparison=ComparisonType.LT,
                # ),
                # _TestExpectation(
                #     expectation=2,
                #     score=CONFIDENCE_THRESHOLD_DEFAULT,
                #     comparison=ComparisonType.LT,
                # ),
            ],
        ),
        (
            "this answer should get a neutral response",
            -1,
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
                _TestExpectation(
                    expectation=0,
                    score=CONFIDENCE_THRESHOLD_DEFAULT,
                    comparison=ComparisonType.LT,
                ),
            ],
        ),
    ],
)
def test_evaluates_for_default_model(
    model_roots: List[str],
    shared_root: str,
    input_answer: str,
    input_expectation_number: int,
    config_data: dict,
    expected_results: List[_TestExpectation],
):
    lesson = "question1-untrained"
    with mocked_data_dao(
        lesson,
        example_data_path(""),
        model_roots[0],
        model_roots[1],
        is_default_model=True,
    ):
        classifier = ClassifierFactory().new_classifier(
            ClassifierConfig(
                dao=opentutor_classifier.dao.find_data_dao(),
                model_name=lesson,
                model_roots=model_roots,
                shared_root=shared_root,
            )
        )
        result = classifier.evaluate(
            AnswerClassifierInput(
                input_sentence=input_answer,
                config_data=dict_to_config(config_data),
                expectation=input_expectation_number,
            )
        )
        assert_classifier_evaluate(result, expected_results)


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,config_data,expected_results,expected_sa_results",
    [
        (
            "I dont know what you are talking about",
            0,
            {},
            [_TestExpectation(expectation=0, score=0.85, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "I do not understand",
            0,
            {},
            [_TestExpectation(expectation=0, score=0.87, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "I believe the answer is peer pressure can change your behavior",
            0,
            {},
            [_TestExpectation(expectation=0, score=0.97, evaluation="Good")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "Fuck you tutor",
            0,
            {},
            [_TestExpectation(expectation=0, score=0.94, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "What the hell is that?",
            0,
            {},
            [_TestExpectation(expectation=0, score=0.94, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "I dont know this shit",
            0,
            {},
            [_TestExpectation(expectation=0, score=0.85, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "I dont know this shit but I guess the answer is peer pressure can change your behavior",
            0,
            {},
            [_TestExpectation(expectation=0, score=0.97, evaluation="Good")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "assistant, assistance",
            0,
            {},
            [_TestExpectation(expectation=0, score=0.94, evaluation="Bad")],
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
    ],
)
@responses.activate
def test_evaluates_meta_cognitive_sentences(
    model_roots,
    shared_root,
    input_answer: str,
    input_expectation_number: int,
    config_data: dict,
    expected_results: List[_TestExpectation],
    expected_sa_results: dict,
):
    lesson = "question1"
    with mocked_data_dao(lesson, example_data_path(""), model_roots[0], model_roots[1]):
        classifier = ClassifierFactory().new_classifier(
            ClassifierConfig(
                dao=opentutor_classifier.dao.find_data_dao(),
                model_name=lesson,
                model_roots=model_roots,
                shared_root=shared_root,
            )
        )
        result = classifier.evaluate(
            AnswerClassifierInput(
                input_sentence=input_answer,
                config_data=dict_to_config(config_data),
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
            expected_sa_results["profanity"].score
            == result.speech_acts["profanity"].score
        )
        assert_classifier_evaluate(result, expected_results)
