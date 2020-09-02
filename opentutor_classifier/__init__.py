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
class ExpectationFeatures:
    ideal: str


@dataclass
class QuestionConfig:
    question: str = ""
    expectation_features: List[ExpectationFeatures] = field(default_factory=list)

    def __post_init__(self):
        self.expectation_features = [
            x if isinstance(x, ExpectationFeatures) else ExpectationFeatures(**x)
            for x in self.expectation_features or []
        ]


@dataclass
class AnswerClassifierInput:
    input_sentence: str
    config_data: Optional[QuestionConfig] = None
    expectation: int = -1


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
    config: dict = field(default_factory=dict)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, encoding="latin-1")


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)
