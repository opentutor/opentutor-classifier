from dataclasses import dataclass


@dataclass
class Answer:
    answer: str


@dataclass
class AnswerEvaluation:
    answer: Answer
    score: float


class Classifier:
    """
    Not necesarily the correct interface for the classifier yet, just a stub example
    """

    def evaluate(self, answer: Answer) -> AnswerEvaluation:
        return AnswerEvaluation(answer=answer, score=0.0)
