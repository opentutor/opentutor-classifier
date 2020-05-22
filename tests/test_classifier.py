import pytest
from opentutor_classifier import Answer, Classifier


@pytest.mark.parametrize(
    "input_answer,expected_score",
    [
        ("some user answer 1", 0.0),
        ("some other answer 2 (this will fail for now)", 1.0),
    ],
)
def test_trained_classifier_evalulates_answers_for_a_hard_coded_question(
    input_answer, expected_score
):
    classifier = (
        Classifier()
    )  # really this should be loaded with model, maybe trained inline for this test?
    result = classifier.evaluate(Answer(answer=input_answer))
    assert result.score == expected_score
