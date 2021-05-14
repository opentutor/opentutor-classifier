from os import path

import opentutor_classifier
from . import (
    ClassifierFactory,
    TrainingConfig,
    TrainingResult,
)
from . import DataDao
from .dao import FileDataDao


def train(
    lesson: str,
    arch="",
    config: TrainingConfig = None,
    dao: DataDao = None,
) -> TrainingResult:
    dao = dao or opentutor_classifier.dao.find_data_dao()
    data = dao.find_training_input(lesson)
    fac = ClassifierFactory()
    training = fac.new_training(config or TrainingConfig(), arch=arch)
    res = training.train(data, dao)
    return res


def train_data_root(
    arch="", config: TrainingConfig = None, data_root="data", output_dir=""
) -> TrainingResult:
    droot, lesson = path.split(path.abspath(data_root))
    return train(
        lesson,
        arch=arch,
        config=config,
        dao=FileDataDao(droot, model_root=output_dir),
    )


def train_default_data_root(
    arch="", config: TrainingConfig = None, data_root="data", output_dir=""
) -> TrainingResult:
    droot, __default__ = path.split(path.abspath(data_root))
    return train_default(
        arch=arch, config=config, dao=FileDataDao(droot, model_root=output_dir)
    )


def train_default(
    arch="",
    config: TrainingConfig = None,
    dao: DataDao = None,
) -> TrainingResult:
    dao = dao or opentutor_classifier.dao.find_data_dao()
    data = dao.find_default_training_data()
    return (
        ClassifierFactory()
        .new_training(config or TrainingConfig(), arch=arch)
        .train_default(data, dao)
    )
