from os import path

from . import (
    ClassifierFactory,
    QuestionConfig,
    TrainingConfig,
    TrainingInput,
    TrainingOptions,
    TrainingResult,
)
from .api import fetch_training_data, fetch_all_training_data
from .utils import load_data, load_yaml


def train_data_root(
    data_root="data",
    arch="",
    config: TrainingConfig = None,
    opts: TrainingOptions = None,
):
    return (
        ClassifierFactory()
        .new_training(config or TrainingConfig(), arch=arch)
        .train(
            TrainingInput(
                config=QuestionConfig(**load_yaml(path.join(data_root, "config.yaml"))),
                data=load_data(path.join(data_root, "training.csv")),
            ),
            opts or TrainingOptions(),
        )
    )


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
    return (
        ClassifierFactory()
        .new_training(config, arch=arch)
        .train(
            fetch_training_data(lesson, url=fetch_training_data_url),
            opts,
        )
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
