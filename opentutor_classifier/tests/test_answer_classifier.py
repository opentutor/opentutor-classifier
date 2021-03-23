#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
from typing import List

import pytest

from opentutor_classifier import (
    ARCH_SVM_CLASSIFIER,
    AnswerClassifierInput,
    ClassifierConfig,
    ClassifierFactory,
    SpeechActClassifierResult,
)
from opentutor_classifier.config import confidence_threshold_default
from opentutor_classifier.utils import dict_to_config
from .utils import (
    assert_classifier_evaluate,
    assert_testset_accuracy,
    fixture_path,
    read_example_testset,
)
from .types import _TestExpectation


CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.fixture(scope="module")
def model_roots() -> List[str]:
    return [fixture_path("models"), fixture_path("models_deployed")]


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return os.path.dirname(word2vec)


@pytest.mark.parametrize(
    "example,arch,confidence_threshold,expected_accuracy",
    [
        ("question1", ARCH_SVM_CLASSIFIER, CONFIDENCE_THRESHOLD_DEFAULT, 1.0),
        ("question2", ARCH_SVM_CLASSIFIER, CONFIDENCE_THRESHOLD_DEFAULT, 1.0),
    ],
)
def test_evaluate_example(
    model_roots,
    shared_root,
    example: str,
    arch: str,
    confidence_threshold: float,
    expected_accuracy: float,
):
    testset = read_example_testset(example, confidence_threshold=confidence_threshold)
    assert_testset_accuracy(
        arch, os.path.join(model_roots[0], example), shared_root, testset
    )


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,config_data,expected_results",
    [
        (
            "they need sunlight",
            -1,
            {
                "question": "how can i grow better plants?",
                "expectations": [
                    # {"ideal": "give them the right amount of water"},
                    {"ideal": "they need sunlight"},
                ],
            },
            [
                # _TestExpectation(expectation=0, evaluation="Bad", score=0.02),
                _TestExpectation(expectation=0, evaluation="Good", score=1.0),
            ],
        ),
        # (
        #     "peer pressure",
        #     -1,
        #     {
        #         "question": "What are the challenges to demonstrating integrity in a group?",
        #         "expectations": [
        #             {
        #                 "ideal": "Peer pressure can cause you to allow inappropriate behavior"
        #             },
        #             {"ideal": "Enforcing the rules can make you unpopular"},
        #         ],
        #     },
        #     [
        #         # _TestExpectation(expectation=0, evaluation="Bad", score=0.01),
        #         _TestExpectation(expectation=1, evaluation="Bad", score=0.17),
        #     ],
        # ),
        # (
        #     "influence from others can change your behavior",
        #     -1,
        #     {
        #         "question": "What are the challenges to demonstrating integrity in a group?",
        #         "expectations": [
        #             {
        #                 "ideal": "Peer pressure can cause you to allow inappropriate behavior"
        #             }
        #         ],
        #     },
        #     [_TestExpectation(expectation=0, evaluation="Bad", score=0.01)],
        # ),
        # (
        #     "hi",
        #     -1,
        #     {
        #         "question": "What are the challenges to demonstrating integrity in a group?",
        #         "expectations": [
        #             {
        #                 "ideal": "Peer pressure can cause you to allow inappropriate behavior"
        #             }
        #         ],
        #     },
        #     [_TestExpectation(expectation=0, evaluation="Bad", score=0.14)],
        # ),
        # (
        #     "some gibberish kjlsdafhalkjfha",
        #     -1,
        #     {
        #         "question": "What are the challenges to demonstrating integrity in a group?",
        #         "expectations": [
        #             {
        #                 "ideal": "Peer pressure can cause you to allow inappropriate behavior"
        #             }
        #         ],
        #     },
        #     [_TestExpectation(expectation=0, evaluation="Bad", score=0.14)],
        # ),
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
    classifier = ClassifierFactory().new_classifier_default(
        ClassifierConfig(
            model_name="default", model_roots=model_roots, shared_root=shared_root
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
def test_evaluates_meta_cognitive_sentences(
    model_roots,
    shared_root,
    input_answer: str,
    input_expectation_number: int,
    config_data: dict,
    expected_results: List[_TestExpectation],
    expected_sa_results: dict,
):
    classifier = ClassifierFactory().new_classifier(
        ClassifierConfig(
            model_name="question1", model_roots=model_roots, shared_root=shared_root
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
        expected_sa_results["profanity"].score == result.speech_acts["profanity"].score
    )
    assert_classifier_evaluate(result, expected_results)
