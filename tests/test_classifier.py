import pytest
from multi_exp_classifier import DispatcherModel


@pytest.mark.parametrize(
    "input_answer,expected_number, expected_score",
    [
        (
            ["rules can make you unpopular"],
            None,
            {1: [-1.0, "Good"], 2: [1.0, "Bad"], 3: [1.0, "Bad"]},
        ),
        (
            ["peer pressure can get you in trouble"],
            1,
            {1: [-0.6666666666666667, "Good"]},
        ),
    ],
)
def test_trained_classifier_evalulates_answers_for_a_hard_coded_question(
    input_answer, expected_number, expected_score
):
    classifier = (
        DispatcherModel()
    )  # really this should be loaded with model, maybe trained inline for this test?
    score = classifier.predict_sentence(input_answer, expected_number)
    assert score == expected_score
