import os
from os import path
from opentutor_classifier import AnswerClassifierInput
from opentutor_classifier.svm import train_classifier, SVMAnswerClassifier


def fixture_path(p: str) -> str:
    return os.path.abspath(os.path.join(".", "tests", "fixtures", p))


def __train_model(tmpdir) -> str:
    test_root = tmpdir.mkdir("test")
    training_data = fixture_path(path.join("data", "training_data.csv"))
    model_root = os.path.join(test_root, "models")
    train_classifier(training_data, model_root=model_root)
    return model_root


def test_outputs_models_at_specified_model_root(tmpdir):
    model_root = __train_model(tmpdir)
    assert path.exists(path.join(model_root, "models"))
    assert path.exists(path.join(model_root, "ideal_answers"))


def test_trained_models_usable_for_inference(tmpdir):
    model_root = __train_model(tmpdir)
    assert os.path.exists(model_root)
    classifier = SVMAnswerClassifier()
    result = classifier.evaluate(AnswerClassifierInput(input_sentence="peer pressure"))
    assert len(result.expectationResults) == 3
