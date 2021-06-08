from os import path

import pytest

from opentutor_classifier import ClassifierFactory, ARCH_LR_CLASSIFIER, TrainingConfig

from .utils import fixture_path, test_env_isolated


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


@pytest.mark.only
@pytest.mark.parametrize(
    "lesson,arch",
    [
        (
            "mixture_toy",
            ARCH_LR_CLASSIFIER,
        ),
    ],
)
def test_data_replication(tmpdir, data_root, shared_root, lesson: str, arch: str):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        fac = ClassifierFactory()
        training = fac.new_training(
            arch=arch, config=TrainingConfig(shared_root=shared_root)
        )
        assert training is not None
