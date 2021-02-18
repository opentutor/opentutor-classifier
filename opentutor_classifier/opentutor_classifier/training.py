from os import path

from . import (
    ClassifierFactory,
    QuestionConfig,
    TrainingConfig,
    TrainingInput,
    TrainingOptions,
    TrainingResult,
)
from .api import fetch_training_data
from .utils import load_data, load_yaml


def train_data_root(
    data_root="data", config: TrainingConfig = None, opts: TrainingOptions = None
):
    return (
        ClassifierFactory()
        .new_training(config or TrainingConfig())
        .train(
            TrainingInput(
                config=QuestionConfig(**load_yaml(path.join(data_root, "config.yaml"))),
                data=load_data(path.join(data_root, "training.csv")),
            ),
            opts or TrainingOptions(),
        )
    )


def train_default(
    data_root="data", config: TrainingConfig = None, opts: TrainingOptions = None
) -> TrainingResult:
    return (
        ClassifierFactory()
        .new_training(config or TrainingConfig())
        .train_default(data_root=data_root, config=config, opts=opts)
    )


def train_online(
    lesson: str,
    config: TrainingConfig,
    opts: TrainingOptions,
    fetch_training_data_url="",
) -> TrainingResult:
    return (
        ClassifierFactory()
        .new_training(config)
        .train(
            fetch_training_data(lesson, url=fetch_training_data_url),
            opts,
        )
    )
