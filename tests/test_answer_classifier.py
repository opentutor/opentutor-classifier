import pytest
from os import path
from typing import Tuple
from opentutor_classifier import (
    AnswerClassifierInput,
    ExpectationClassifierResult,
    load_word2vec_model,
    load_question,
)
from opentutor_classifier.svm import SVMAnswerClassifier, load_instances
from . import fixture_path, fixture_path_word2vec_model, fixture_path_question


@pytest.fixture(scope="module")
def model_and_ideal_answers() -> Tuple[dict, dict]:
    return load_instances(fixture_path("models"))


@pytest.fixture(scope="module")
def word2vec_model():
    return load_word2vec_model(
        fixture_path_word2vec_model(path.join("model_word2vec", "model.bin"))
    )


@pytest.fixture(scope="module")
def question():
    return load_question(fixture_path_question(path.join("data", "config.yml")))


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,expected_results",
    [
        (
            "peer pressure can change your behavior",
            0,
            [ExpectationClassifierResult(expectation=0, score=0.94, evaluation="Good")],
        )
    ],
)
def test_evaluates_one_expectation(
    question,
    word2vec_model,
    model_and_ideal_answers,
    input_answer,
    input_expectation_number,
    expected_results,
):
    model_instances, ideal_answers = model_and_ideal_answers

    classifier = SVMAnswerClassifier(
        model_instances, ideal_answers, word2vec_model, question
    )
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer, expectation=input_expectation_number
        )
    )
    assert len(result.expectation_results) == len(expected_results)
    for res, res_expected in zip(result.expectation_results, expected_results):
        assert round(res.score, 2) == res_expected.score
        assert res.evaluation == res_expected.evaluation


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,expected_results",
    [
        (
            "peer pressure can change your behavior",
            -1,
            [
                ExpectationClassifierResult(
                    expectation=0, score=0.94, evaluation="Good"
                ),
                ExpectationClassifierResult(
                    expectation=1, score=0.23, evaluation="Bad"
                ),
                ExpectationClassifierResult(
                    expectation=2, score=0.28, evaluation="Bad"
                ),
            ],
        )
    ],
)
def test_evaluates_with_no_input_expectation_number(
    question,
    word2vec_model,
    model_and_ideal_answers,
    input_answer,
    input_expectation_number,
    expected_results,
):
    model_instances, ideal_answers = model_and_ideal_answers
    classifier = SVMAnswerClassifier(
        model_instances, ideal_answers, word2vec_model, question
    )
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer, expectation=input_expectation_number
        )
    )
    assert len(result.expectation_results) == len(expected_results)
    for res, res_expected in zip(result.expectation_results, expected_results):
        assert round(res.score, 2) == res_expected.score
        assert res.evaluation == res_expected.evaluation
