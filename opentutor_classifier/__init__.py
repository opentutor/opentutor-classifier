from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
from typing import List


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
    expectationResults: List[ExpectationClassifierResult] = field(default_factory=list)


class AnswerClassifier(ABC):
    @abstractmethod
    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        raise NotImplementedError()


def loadData(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, encoding="latin-1")
