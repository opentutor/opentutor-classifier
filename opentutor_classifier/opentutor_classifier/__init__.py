#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from importlib import import_module
from os import environ
import pandas as pd
from typing import Any, Dict, List, Optional
import yaml
from opentutor_classifier.speechact import SpeechActClassifierResult


@dataclass
class ExpectationClassifierResult:
    expectation: int = -1
    evaluation: str = ""
    score: float = 0.0


@dataclass
class ExpectationConfig:
    ideal: str = ""
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionConfig:
    question: str = ""
    expectations: List[ExpectationConfig] = field(default_factory=list)

    def __post_init__(self):
        self.expectations = [
            x if isinstance(x, ExpectationConfig) else ExpectationConfig(**x)
            for x in self.expectations or []
        ]

    def get_expectation_feature(
        self, exp: int, feature_name: str, dft: Any = None
    ) -> Any:
        return (
            self.expectations[exp].features.get(feature_name, dft)
            if exp >= 0 and exp < len(self.expectations)
            else dft
        )

    def get_expectation_ideal(self, exp: int) -> Any:
        return (
            self.expectations[exp].ideal
            if exp >= 0 and exp < len(self.expectations)
            else ""
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def write_to(self, file_path: str):
        with open(file_path, "w") as config_file:
            yaml.safe_dump(self.to_dict(), config_file)


@dataclass
class AnswerClassifierInput:
    input_sentence: str
    config_data: Optional[QuestionConfig] = None
    expectation: int = -1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnswerClassifierResult:
    input: AnswerClassifierInput
    expectation_results: List[ExpectationClassifierResult] = field(default_factory=list)
    speech_acts: Dict[str, SpeechActClassifierResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExpectationTrainingResult:
    accuracy: float = 0


@dataclass
class ClassifierConfig:
    model_name: str
    shared_root: str = "shared"
    model_roots: List[str] = field(
        default_factory=lambda: ["models", "models_deployed"]
    )


class AnswerClassifier(ABC):
    @abstractmethod
    def configure(self, answer: ClassifierConfig) -> "AnswerClassifier":
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        raise NotImplementedError()


@dataclass
class TrainingResult:
    lesson: str = ""  # the lesson id
    expectations: List[ExpectationTrainingResult] = field(default_factory=list)
    models: str = ""  # path to directory with newly trained models for this lesson
    archive: str = ""  # path to directory where previously active models for this lesson (if any) have been archived

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class TrainingOptions:
    add_ideal_answers_to_training_data: bool = True
    archive_root: str = "archive"
    output_dir: str = "output"


@dataclass
class TrainingInput:
    lesson: str = ""  # the lesson id
    config: QuestionConfig = field(default_factory=QuestionConfig)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = QuestionConfig(**self.config)


@dataclass
class TrainingConfig:
    shared_root: str = "shared"


class AnswerClassifierTraining(ABC):
    @abstractmethod
    def configure(self, config: TrainingConfig) -> "AnswerClassifierTraining":
        raise NotImplementedError()

    @abstractmethod
    def train(
        self, train_input: TrainingInput, config: TrainingOptions
    ) -> TrainingResult:
        raise NotImplementedError()

    @abstractmethod
    def train_default(
        self,
        data_root: str = "data",
        config: TrainingConfig = None,
        opts: TrainingOptions = None,
    ) -> TrainingResult:
        raise NotImplementedError()


def dict_to_question_config(d: Dict[str, Any]) -> QuestionConfig:
    return QuestionConfig(
        question=d.get("question") or "",
        expectations=[
            ExpectationConfig(
                ideal=x.get("ideal") or "", features=x.get("features") or {}
            )
            for x in d.get("expectations") or []
        ],
    )


def load_question_config(f: str) -> QuestionConfig:
    with open(f, "r") as yaml_file:
        return dict_to_question_config(yaml.load(yaml_file, Loader=yaml.FullLoader))


class ArchClassifierFactory(ABC):
    @abstractmethod
    def new_classifier(self, config: ClassifierConfig) -> AnswerClassifier:
        raise NotImplementedError()

    @abstractmethod
    def new_classifier_default(self, config: ClassifierConfig) -> AnswerClassifier:
        raise NotImplementedError()

    @abstractmethod
    def new_training(self, config: TrainingConfig) -> AnswerClassifierTraining:
        raise NotImplementedError()


_factories_by_arch: Dict[str, ArchClassifierFactory] = {}


def register_classifier_factory(arch: str, fac: ArchClassifierFactory) -> None:
    _factories_by_arch[arch] = fac


ARCH_SVM_CLASSIFIER = "opentutor_classifier.svm"
ARCH_LR_CLASSIFIER = "opentutor_classifier.lr"
ARCH_DEFAULT = "opentutor_classifier.svm"


class ClassifierFactory:
    def _find_arch_fac(self, arch: str) -> ArchClassifierFactory:
        arch = arch or environ.get("CLASSIFIER_ARCH") or ARCH_DEFAULT
        if arch not in _factories_by_arch:
            import_module(arch)
        f = _factories_by_arch[arch]
        return f

    def new_classifier(self, config: ClassifierConfig, arch="") -> AnswerClassifier:
        return self._find_arch_fac(arch).new_classifier(config)

    def new_classifier_default(
        self, config: ClassifierConfig, arch=""
    ) -> AnswerClassifier:
        return self._find_arch_fac(arch).new_classifier_default(config)

    def new_training(self, config: TrainingConfig, arch="") -> AnswerClassifierTraining:
        return self._find_arch_fac(arch).new_training(config)
