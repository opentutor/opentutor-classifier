# import pytest
# from opentutor_classifier import AnswerClassifierInput, ExpectationClassifierResult
# from opentutor_classifier.svm import SVMAnswerClassifier, load_config_into_objects
# from . import fixture_path
# import os


# @pytest.fixture(scope="module")
# def model_root() -> str:
#     return fixture_path("models")


# @pytest.fixture(scope="module")
# def shared_root() -> str:
#     return fixture_path("shared")


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
                ExpectationClassifierResult(expectation=0, evaluation="Bad", score=0.01),
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
