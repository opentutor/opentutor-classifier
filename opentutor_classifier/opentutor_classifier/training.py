from os import path

import opentutor_classifier
from . import (
    ClassifierFactory,
    DataDao,
    TrainingConfig,
    TrainingOptions,
    TrainingResult,
)
from .api import FileDataDao, fetch_all_training_data


def train(
    dao: DataDao,
    lesson: str,
    config: TrainingConfig = None,
    opts: TrainingOptions = None,
    arch="",
) -> TrainingResult:
    data = dao.find_training_input(lesson)
    fac = ClassifierFactory()
    training = fac.new_training(config or TrainingConfig(), arch=arch)
    res = training.train(data, opts or TrainingOptions())
    return res


def train_data_root(
    arch="",
    config: TrainingConfig = None,
    data_root="data",
    opts: TrainingOptions = None,
) -> TrainingResult:
    droot, lesson = path.split(path.abspath(data_root))
    return train(FileDataDao(droot), lesson, config=config, opts=opts, arch=arch)


def train_default(
    data_root="data",
    arch="",
    config: TrainingConfig = None,
    opts: TrainingOptions = None,
) -> TrainingResult:
    return (
        ClassifierFactory()
        .new_training(config or TrainingConfig(), arch=arch)
        .train_default(data_root=data_root, config=config, opts=opts)
    )


def train_online(
    lesson: str,
    config: TrainingConfig,
    opts: TrainingOptions,
    arch="",
    fetch_training_data_url="",
) -> TrainingResult:
    return train(
        opentutor_classifier.find_data_dao(),
        lesson,
        config=config,
        opts=opts,
        arch=arch,
    )


def train_default_online(
    config: TrainingConfig,
    opts: TrainingOptions,
    arch="",
    fetch_training_data_url="",
) -> TrainingResult:
    return (
        ClassifierFactory()
        .new_training(config, arch=arch)
        .train_default_online(
            fetch_all_training_data(url=fetch_training_data_url),
            config,
            opts,
        )
    )
