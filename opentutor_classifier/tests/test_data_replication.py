from os import path

import pytest
import pandas as pd
import responses

from opentutor_classifier.config import confidence_threshold_default


from opentutor_classifier import (
    ClassifierFactory,
    ARCH_LR_CLASSIFIER,
    TrainingConfig,
    TrainingInput,
)
from opentutor_classifier.dao import (
    find_data_dao,
    FileDataDao,
    load_data,
    load_config,
    _CONFIG_YAML,
    _TRAINING_CSV,
)

from .utils import (
    fixture_path, 
    test_env_isolated,  
    assert_inc_testset_accuracy,
    read_example_testset,
    assert_testset_accuracy
)

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()

@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


@responses.activate
@pytest.mark.only
@pytest.mark.parametrize(
    "lesson,arch,confidence_threshold",
    [
        (
            "mixture_toy",
            ARCH_LR_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
        ),
    ],
)
def test_data_replication(tmpdir, data_root, shared_root, lesson: str, arch: str,confidence_threshold:float):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson,
    ) as test_config:
        rep_factor = [1, 2, 5, 10]
        data = load_data(path.join(data_root, lesson, _TRAINING_CSV))  # dao.py
        results = []
        for i in rep_factor:
            dao = FileDataDao(data_root, model_root = test_config.output_dir)
            data_list = [data] * i
            new_data = pd.concat(data_list)
            input = TrainingInput(
                lesson=lesson,
                config=load_config(path.join(data_root, lesson, _CONFIG_YAML)),
                data=new_data,  # dataframe
            )
            fac = ClassifierFactory()
            training = fac.new_training(
            arch=arch, config=TrainingConfig(shared_root=shared_root)
            )
            train_result = training.train(input, dao)
            results.append(train_result)
        testset = read_example_testset(
            lesson, confidence_threshold=confidence_threshold
        )
        assert_inc_testset_accuracy(
            arch,
            results,
            shared_root,
            testset,
        )
