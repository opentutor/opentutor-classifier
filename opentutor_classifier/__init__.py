#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
import pandas as pd
from typing import Any, Dict, List
import yaml
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
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
    question: str
    expectation_features: List[ExpectationFeatures]


@dataclass
class AnswerClassifierInput:
    input_sentence: str
    config_data: QuestionConfig
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


@dataclass
class TrainingResult:
    lesson: str = ""
    expectations: List[ExpectationTrainingResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class AnswerClassifier(ABC):
    @abstractmethod
    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        raise NotImplementedError()


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, encoding="latin-1")


def load_word2vec_model(path: str) -> Word2VecKeyedVectors:
    return KeyedVectors.load_word2vec_format(path, binary=True)


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)
