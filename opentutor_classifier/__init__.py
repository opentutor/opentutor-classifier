from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
from typing import List
import yaml
from gensim.models import KeyedVectors
from typing import Any


@dataclass
class ExpectationClassifierResult:
    expectation: int = -1
    evaluation: str = ""
    score: float = 0.0


@dataclass
class AnswerClassifierInput:
    input_sentence: str
    expectation: int = -1


@dataclass
class AnswerClassifierResult:
    input: AnswerClassifierInput
    expectation_results: List[ExpectationClassifierResult] = field(default_factory=list)


class AnswerClassifier(ABC):
    @abstractmethod
    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        raise NotImplementedError()


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, encoding="latin-1")


def load_word2vec_model(path: str) -> Any:
    return KeyedVectors.load_word2vec_format(path, binary=True)


def load_question(path: str) -> str:
    file = open(path)
    parsed_file = yaml.load(file, Loader=yaml.FullLoader)
    return parsed_file["main_question"]
