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
import json
import asyncio
from unittest.mock import patch
from opentutor_classifier import (
    get_classifier_arch,
    AnswerClassifier,
    AnswerClassifierInput,
    ClassifierConfig,
    ClassifierFactory,
    SpeechActClassifierResult,
    TrainingConfig,
    ARCH_OPENAI_CLASSIFIER,
    ARCH_COMPOSITE_CLASSIFIER,
)
from opentutor_classifier.config import confidence_threshold_default, EVALUATION_BAD
import opentutor_classifier.dao
from opentutor_classifier.log import logger
from opentutor_classifier.training import train_data_root, train_default_data_root
from opentutor_classifier.utils import dict_to_config
from .utils import (
    assert_classifier_evaluate,
    assert_testset_accuracy,
    fixture_path,
    example_data_path,
    mocked_data_dao,
    read_example_testset,
    mock_openai_object,
    mock_openai_timeout,
)
from .types import ComparisonType, _TestExpectation

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.fixture(scope="module")
def model_roots() -> List[str]:
    return [
        fixture_path("models"),
        fixture_path("models_deployed"),
        fixture_path("data"),
    ]


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return os.path.dirname(word2vec)


def _find_or_train_classifier(
    lesson: str, model_root: str, data_root: str, shared_root: str, arch=""
) -> AnswerClassifier:
    arch = arch or get_classifier_arch()
    dao = opentutor_classifier.dao.FileDataDao(
        data_root=data_root, model_root=model_root
    )

    cfac = ClassifierFactory()
    cconf = ClassifierConfig(
        dao=dao,
        model_name=lesson,
        model_roots=[model_root],
        shared_root=shared_root,
    )
    if not cfac.has_trained_model(lesson, cconf, arch=arch):
        example_dir = os.path.join(data_root, lesson)
        logger.warning(
            f"trained model not found in fixtures for test lesson {lesson}, attempting to train..."
        )
        if lesson == "default":
            train_default_data_root(
                data_root=example_dir,
                config=TrainingConfig(shared_root=shared_root),
                output_dir=model_root,
                arch=arch,
            )
        else:
            train_data_root(
                data_root=example_dir,
                config=TrainingConfig(shared_root=shared_root),
                output_dir=model_root,
                arch=arch,
            )
    return cfac.new_classifier(cconf, arch=arch)


@pytest.mark.parametrize(
    "lesson,arch,input_answer,config_data,mock_payload",
    [
        (
            "candles",
            ARCH_COMPOSITE_CLASSIFIER,
            "The penguins are full of fish",
            {
                "question": "What are the challenges to demonstrating integrity in a group?",
                "expectations": [
                    {
                        "expectation_id": "0",
                        "ideal": "Peer pressure can cause you to allow inappropriate behavior",
                    },
                    {
                        "expectation_id": "1",
                        "ideal": "Enforcing the rules can make you unpopular",
                    },
                ],
            },
            {
                "answers": {
                    "answer_1": {
                        "answer text": "it explodes",
                        "concepts": {
                            "concept_0": {
                                "is_known": "false",
                                "confidence": 0.9,
                                "justification": "The answer does not mention anything about normal diodes not conducting current in reverse bias, which is the concept being tested.",
                            },
                            "concept_1": {
                                "is_known": "true",
                                "confidence": 0.7,
                                "justification": "The answer mentions that the diode explodes, which implies that it goes into breakdown mode and gets damaged.",
                            },
                        },
                    }
                }
            },
        ),
    ],
)
def test_composite_answer_classifier_json_response(
    model_roots,
    shared_root,
    lesson: str,
    arch: str,
    input_answer: str,
    config_data: dict,
    mock_payload: dict,
):
    os.environ["OPENAI_API_KEY"] = "fake"
    with mocked_data_dao(lesson, example_data_path(""), model_roots[0], model_roots[1]):
        classifier = _find_or_train_classifier(
            lesson, model_roots[0], model_roots[2], shared_root, arch=arch
        )
        with patch("openai.ChatCompletion.acreate") as mock_create:
            mock_create.side_effect = mock_openai_timeout(json.dumps(mock_payload))
            result = asyncio.run(
                classifier.evaluate(
                    AnswerClassifierInput(
                        input_sentence=input_answer,
                        config_data=dict_to_config(config_data),
                    )
                )
            )
        print(json.dumps(result.to_dict(), indent=2))
        assert result.expectation_results[0].evaluation == EVALUATION_BAD


@pytest.mark.parametrize(
    "lesson,arch,input_answer,config_data,mock_payload",
    [
        (
            "candles",
            ARCH_OPENAI_CLASSIFIER,
            "The penguins are full of fish",
            {
                "question": "What are the challenges to demonstrating integrity in a group?",
                "expectations": [
                    {
                        "expectation_id": "0",
                        "ideal": "Peer pressure can cause you to allow inappropriate behavior",
                    },
                    {
                        "expectation_id": "1",
                        "ideal": "Enforcing the rules can make you unpopular",
                    },
                ],
            },
            {
                "answers": {
                    "answer_1": {
                        "answer text": "it explodes",
                        "concepts": {
                            "concept_0": {
                                "is_known": "false",
                                "confidence": 0.9,
                                "justification": "The answer does not mention anything about normal diodes not conducting current in reverse bias, which is the concept being tested.",
                            },
                            "concept_1": {
                                "is_known": "true",
                                "confidence": 0.7,
                                "justification": "The answer mentions that the diode explodes, which implies that it goes into breakdown mode and gets damaged.",
                            },
                        },
                    }
                }
            },
        ),
    ],
)
def test_openai_answer_classifier_json_response(
    model_roots,
    shared_root,
    lesson: str,
    arch: str,
    input_answer: str,
    config_data: dict,
    mock_payload: dict,
):
    os.environ["OPENAI_API_KEY"] = "fake"
    with mocked_data_dao(lesson, example_data_path(""), model_roots[0], model_roots[1]):
        classifier = _find_or_train_classifier(
            lesson, model_roots[0], model_roots[2], shared_root, arch=arch
        )
        with patch("openai.ChatCompletion.acreate") as mock_create:
            mock_create.return_value = mock_openai_object(json.dumps(mock_payload))
            result = asyncio.run(
                classifier.evaluate(
                    AnswerClassifierInput(
                        input_sentence=input_answer,
                        config_data=dict_to_config(config_data),
                    )
                )
            )
        print(json.dumps(result.to_dict(), indent=2))
        assert result.expectation_results[0].evaluation == EVALUATION_BAD


@pytest.mark.parametrize(
    "lesson,arch,confidence_threshold,expected_accuracy",
    [
        ("question1", "", CONFIDENCE_THRESHOLD_DEFAULT, 0.66),
        ("question2", "", CONFIDENCE_THRESHOLD_DEFAULT, 1.0),
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
        _find_or_train_classifier(
            lesson, model_roots[0], model_roots[2], shared_root, arch=arch
        )
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
            "",
            {
                "question": "What are the challenges to demonstrating integrity in a group?",
                "expectations": [
                    {
                        "expectation_id": "0",
                        "ideal": "Peer pressure can cause you to allow inappropriate behavior",
                    },
                    {
                        "expectation_id": "1",
                        "ideal": "Enforcing the rules can make you unpopular",
                    },
                ],
            },
            [
                _TestExpectation(expectation="0", evaluation="Good", score=0.65),
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
            "",
            {
                "question": "What are the challenges to demonstrating integrity in a group?",
                "expectations": [
                    {
                        "expectation_id": "0",
                        "ideal": "Peer pressure can cause you to allow inappropriate behavior",
                    },
                    {
                        "expectation_id": "1",
                        "ideal": "Enforcing the rules can make you unpopular",
                    },
                ],
            },
            [
                _TestExpectation(
                    expectation="0",
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
    input_expectation_number: str,
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
        _find_or_train_classifier(
            "default", model_roots[0], model_roots[2], shared_root
        )
        classifier = ClassifierFactory().new_classifier(
            ClassifierConfig(
                dao=opentutor_classifier.dao.find_data_dao(),
                model_name=lesson,
                model_roots=model_roots,
                shared_root=shared_root,
            )
        )
        result = asyncio.run(
            classifier.evaluate(
                AnswerClassifierInput(
                    input_sentence=input_answer,
                    config_data=dict_to_config(config_data),
                    expectation=input_expectation_number,
                )
            )
        )
        assert_classifier_evaluate(result, expected_results)


@pytest.mark.parametrize(
    "lesson, arch, input_answer,input_expectation_number,config_data,expected_sa_results",
    [
        (
            "question1",
            "",
            "I dont know what you are talking about",
            "0",
            {},
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "question1",
            "",
            "I do not understand",
            "0",
            {},
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "question1",
            "",
            "I believe the answer is peer pressure can change your behavior",
            "0",
            {},
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Bad", score=0),
            },
        ),
        (
            "question1",
            "",
            "Fuck you tutor",
            "0",
            {},
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "question1",
            "",
            "What the hell is that?",
            "0",
            {},
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Bad", score=0),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "question1",
            "",
            "I dont know this shit",
            "0",
            {},
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "question1",
            "",
            "I dont know this shit but I guess the answer is peer pressure can change your behavior",
            "0",
            {},
            {
                "metacognitive": SpeechActClassifierResult(evaluation="Good", score=1),
                "profanity": SpeechActClassifierResult(evaluation="Good", score=1),
            },
        ),
        (
            "question1",
            "",
            "assistant, assistance",
            "0",
            {},
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
    lesson: str,
    arch: str,
    input_answer: str,
    input_expectation_number: str,
    config_data: dict,
    expected_sa_results: dict,
):
    with mocked_data_dao(lesson, example_data_path(""), model_roots[0], model_roots[1]):
        classifier = _find_or_train_classifier(
            lesson, model_roots[0], model_roots[2], shared_root, arch=arch
        )
        result = asyncio.run(
            classifier.evaluate(
                AnswerClassifierInput(
                    input_sentence=input_answer,
                    config_data=dict_to_config(config_data),
                    expectation=input_expectation_number,
                )
            )
        )
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
