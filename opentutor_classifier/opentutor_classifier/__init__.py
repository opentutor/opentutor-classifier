#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
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


class AnswerClassifier(ABC):
    @abstractmethod
    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        raise NotImplementedError()


@dataclass
class TrainingResult:
    lesson: str = ""
    expectations: List[ExpectationTrainingResult] = field(default_factory=list)
    models: str = ""
    archive: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class TrainingInput:
    lesson: str = ""
    config: QuestionConfig = field(default_factory=QuestionConfig)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = QuestionConfig(**self.config)


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
