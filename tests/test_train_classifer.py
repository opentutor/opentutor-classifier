import os
from os import path
from opentutor_classifier import AnswerClassifierInput, load_word2vec_model
from opentutor_classifier.svm import train_classifier, SVMAnswerClassifier
from typing import Dict, Tuple
from . import fixture_path, fixture_path_word2vec_model


def __train_model(tmpdir) -> Tuple[str, Dict[int, int]]:
    test_root = tmpdir.mkdir("test")
    training_data = fixture_path(path.join("data", "lesson1_dataset.csv"))
    question_path = fixture_path(path.join("data", "config.yaml"))
    word2vec_model_path = fixture_path_word2vec_model(
        path.join("model_word2vec", "model.bin")
    )
    model_root = os.path.join(test_root, "model_root")
    accuracy = train_classifier(
        question_path, training_data, word2vec_model_path, model_root=model_root
    )
    return model_root, accuracy


def test_outputs_models_at_specified_model_root(tmpdir):
    model_root, _ = __train_model(tmpdir)
    assert path.exists(path.join(model_root, "models_by_expectation_num.pkl"))
    assert path.exists(path.join(model_root, "ideal_answers_by_expectation_num.pkl"))
    assert path.exists(path.join(model_root, "config.yaml"))


def test_trained_models_usable_for_inference(tmpdir):
    model_root, accuracy = __train_model(tmpdir)
    assert os.path.exists(model_root)
    for model_num, acc in accuracy.items():
        if model_num == 0:
            assert acc == 90.0
        if model_num == 1:
            assert acc == 70.0
        if model_num == 2:
            assert acc == 70.0
    word2vec_model = load_word2vec_model(
        fixture_path_word2vec_model(path.join("model_word2vec", "model.bin"))
    )
    classifier = SVMAnswerClassifier(model_root, word2vec_model)
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
