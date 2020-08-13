from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
import pandas as pd
from typing import Any, Dict, List
import yaml
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors


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
class SpeechActClassifierResult:
    evaluation: str = ""
    score: float = 0.0

@dataclass
class AnswerClassifierResult:
    input: AnswerClassifierInput
    expectation_results: List[ExpectationClassifierResult] = field(default_factory=list)
    speech_acts: Dict[str, SpeechActClassifierResult]  = field(default_factory=dict)

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
