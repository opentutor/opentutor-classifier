from typing import Any, Dict, List
from os import path

import pytest

from opentutor_classifier import ARCH_LR_CLASSIFIER
from opentutor_classifier.utils import load_config
from .utils import fixture_path, test_env_isolated, train_classifier


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


# @pytest.mark.only
@pytest.mark.parametrize(
    "lesson,expected_features",
    [
        (
            "lr-feature-gen-1",
            [
                {
                    "bad": [],
                    "good": [],
                    "patterns_bad": [],
                    "patterns_good": [
                        "peer",
                        "peer + pressure",
                        "pressure",
                    ],
                }
            ],
        )
    ],
)
def test_generates_features(
    lesson: str,
    expected_features: List[Dict[str, Any]],
    tmpdir,
    data_root: str,
    shared_root: str,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=ARCH_LR_CLASSIFIER, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)
        config_file = path.join(test_config.data_root, lesson, "config.yaml")
        assert path.isfile(config_file)
        generated_config = load_config(config_file)
        assert [e.features for e in generated_config.expectations] == expected_features
