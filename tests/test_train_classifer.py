import os
from os import path
from opentutor_classifier import (
    AnswerClassifierInput,
    load_question,
    load_word2vec_model,
)
from opentutor_classifier.svm import (
    train_classifier,
    SVMAnswerClassifier,
    load_instances,
)
from . import fixture_path, fixture_path_word2vec_model, fixture_path_question


def __train_model(tmpdir) -> str:
    test_root = tmpdir.mkdir("test")
    training_data = fixture_path(path.join("data", "lesson1_dataset.csv"))
    question_path = fixture_path_question(path.join("data", "config.yml"))
    word2vec_model_path = fixture_path_word2vec_model(
        path.join("model_word2vec", "model.bin")
    )
    model_root = os.path.join(test_root, "model_instances")
    accuracy = train_classifier(
        question_path, training_data, word2vec_model_path, model_root=model_root
    )
    for model_num, acc in accuracy.items():
        if model_num == 0:
            assert acc == 90.0
        if model_num == 1:
            assert acc == 70.0
        if model_num == 2:
            assert acc == 70.0
    return model_root


def test_outputs_models_at_specified_model_root(tmpdir):
    model_root = __train_model(tmpdir)
    assert path.exists(path.join(model_root, "model_instances"))
    assert path.exists(path.join(model_root, "ideal_answers"))


def test_trained_models_usable_for_inference(tmpdir):
    model_root = __train_model(tmpdir)
    assert os.path.exists(model_root)
    model_instances, ideal_answers = load_instances(model_root=model_root)
    word2vec_model = load_word2vec_model(
        fixture_path_word2vec_model(path.join("model_word2vec", "model.bin"))
    )
    question = load_question(fixture_path_question(path.join("data", "config.yml")))
    print("question = ", question)
    classifier = SVMAnswerClassifier(
        model_instances, ideal_answers, word2vec_model, question
    )
    result = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence="peer pressure can change your behavior", expectation=-1
        )
    )
    assert len(result.expectation_results) == 3
    for exp_res in result.expectation_results:
        if exp_res.expectation == 0:
            assert exp_res.evaluation == "Good"
            assert round(exp_res.score, 2) == 0.94
        if exp_res.expectation == 1:
            assert exp_res.evaluation == "Bad"
            assert round(exp_res.score, 2) == 0.23
        if exp_res.expectation == 2:
            assert exp_res.evaluation == "Bad"
            assert round(exp_res.score, 2) == 0.28
