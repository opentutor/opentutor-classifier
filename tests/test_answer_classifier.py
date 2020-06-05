import pytest
from opentutor_classifier import AnswerClassifierInput, ExpectationClassifierResult
from opentutor_classifier.svm import SVMAnswerClassifier


@pytest.mark.parametrize(
    "input_answer,input_expectation,expected_results",
    [
        (
            ["peer pressure can get you in trouble"],
            1,
            [
                ExpectationClassifierResult(
                    expectation=1, score=-0.6666666666666667, evaluation="Good"
                )
            ],
        )
    ],
)
def test_evaluates_one_expectation(input_answer, input_expectation, expected_results):
    classifier = (
        SVMAnswerClassifier()
    )  # really this should be loaded with model, maybe trained inline for this test?
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
                    expectation=1, score=-0.6666666666666667, evaluation="Good"
                ),
                ExpectationClassifierResult(expectation=2, score=1.0, evaluation="Bad"),
                ExpectationClassifierResult(expectation=3, score=1.0, evaluation="Bad"),
            ],
        )
    ],
)
def test_evaluates_with_no_input_expectation(
    input_answer, input_expectation_number, expected_results
):
    classifier = (
        SVMAnswerClassifier()
    )  # really this should be loaded with model, maybe trained inline for this test?
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_answer, expectation=input_expectation_number
        )
    )
    assert len(result.expectationResults) == len(expected_results)
    for res, res_expected in zip(result.expectationResults, expected_results):
        assert res.score == res_expected.score
        assert res.evaluation == res_expected.evaluation
