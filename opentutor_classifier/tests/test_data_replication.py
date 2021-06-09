from os import path

import pytest
import pandas as pd
import responses

from opentutor_classifier import (
    ClassifierFactory,
    ARCH_LR_CLASSIFIER,
    TrainingConfig,
    TrainingInput,
)
from opentutor_classifier.dao import (
    find_data_dao,
    load_data,
    load_config,
    _CONFIG_YAML,
    _TRAINING_CSV,
)

from .utils import fixture_path, test_env_isolated


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


@responses.activate
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
        rep_factor = [1, 2, 5, 10]
        data = load_data(path.join(data_root, lesson, _TRAINING_CSV))  # dao.py
        dao = find_data_dao()
        for i in rep_factor:
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
        assert training is not None


# train_result = train_classifier(lesson, test_config)

# return train_data_root(
#         data_root=path.join(data_root, lesson),
#         config=TrainingConfig(shared_root=shared_root),
#         output_dir=output_dir,
#         arch=arch,
#     )
#     def train_data_root(
#     arch="", config: TrainingConfig = None, data_root="data", output_dir=""
# ) -> TrainingResult:
#     droot, lesson = path.split(path.abspath(data_root))
#     return train(
#         lesson,
#         arch=arch,
#         config=config,
#         dao=FileDataDao(data_root, model_root=output_dir),
#     )

#   def __init__(
#         self, data_root: str, model_root: str = "", deployed_model_root: str = ""
#     ):
#         self._data_root = data_root
#         self._model_root = output_dir
#         self._deployed_model_root = ""

# def train(
#     lesson: str,
#     arch="",
#     config: TrainingConfig = None,
#     dao: DataDao = None,
# ) -> TrainingResult:
#     dao = dao or opentutor_classifier.dao.find_data_dao()
#     data = dao.find_training_input(lesson)
#     fac = ClassifierFactory()
#     training = fac.new_training(config or TrainingConfig(), arch=arch)
#     res = training.train(data, dao)
#     return res


#     def find_training_input(self, lesson: str) -> TrainingInput:
#         return TrainingInput(
#             lesson=lesson,
#             config=self.find_training_config(lesson),
#             data=load_data(self._get_data_file(lesson, _TRAINING_CSV)),
#         )