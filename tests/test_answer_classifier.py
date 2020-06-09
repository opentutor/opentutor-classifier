import pytest
from typing import Tuple
from opentutor_classifier import AnswerClassifierInput, ExpectationClassifierResult
from opentutor_classifier.svm import SVMAnswerClassifier, load_instances
from . import fixture_path


@pytest.fixture(scope="module")
def model_and_ideal_answers() -> Tuple[dict, dict]:
    return load_instances(fixture_path("models"))


@pytest.mark.parametrize(
    "input_answer,input_expectation,expected_results",
    [
        (
            ["peer pressure can get you in trouble"],
            0,
            [
                ExpectationClassifierResult(
                    expectation=0, score=-0.6666666666666667, evaluation="Good"
                )
            ],
        )
    ],
)
def test_evaluates_one_expectation(
    model_and_ideal_answers, input_answer, input_expectation, expected_results
):
    model_instances, ideal_answers = model_and_ideal_answers
    classifier = SVMAnswerClassifier(model_instances, ideal_answers)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer, expectation=input_expectation
        )
    )
    assert len(result.expectationResults) == len(expected_results)
    for res, res_expected in zip(result.expectationResults, expected_results):
        assert res.score == res_expected.score
        assert res.evaluation == res_expected.evaluation


@pytest.mark.parametrize(
    "input_answer,input_expectation_number,expected_results",
    [
        (
            ["peer pressure"],
            -1,
            [
                ExpectationClassifierResult(
                    expectation=0, score=-0.6666666666666667, evaluation="Good"
                ),
                ExpectationClassifierResult(expectation=1, score=1.0, evaluation="Bad"),
                ExpectationClassifierResult(expectation=2, score=1.0, evaluation="Bad"),
            ],
        )
    ],
)
def test_evaluates_with_no_input_expectation(
    model_and_ideal_answers, input_answer, input_expectation_number, expected_results
):
    model_instances, ideal_answers = model_and_ideal_answers
    classifier = SVMAnswerClassifier(model_instances, ideal_answers)
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer, expectation=input_expectation_number
        )
    )
    assert len(result.expectationResults) == len(expected_results)
    for res, res_expected in zip(result.expectationResults, expected_results):
        assert res.score == res_expected.score
        assert res.evaluation == res_expected.evaluation
