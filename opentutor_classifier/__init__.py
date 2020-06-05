from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExpectationClassifierResult:
    # answer: str = ""
    expectation: int = 0
    evaluation: str = ""
    score: float = 0.0


@dataclass
class AnswerClassifierInput:
    input_sentence: str
    expectation: int


@dataclass
class AnswerClassifierResult:
    input: AnswerClassifierInput
    expectationResults: List[ExpectationClassifierResult] = field(default_factory=list)


class AnswerClassifier(ABC):
    @abstractmethod
    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        raise NotImplementedError()
