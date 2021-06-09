#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Tuple

from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ExpectationClassifierResult,
)


class ComparisonType(Enum):
    EQ = 0
    GTE = 1
    LT = 2


@dataclass
class _TestConfig:
    data_root: str
    deployed_models: str
    output_dir: str
    shared_root: str
    arch: str = ""
    is_default_model: bool = False


@dataclass
class _TestExpectation:
    expectation: int = -1  # if not -1, which lesson expectation by index?
    evaluation: str = ""  # Good | Bad
    score: float = 0.0  # confidence
    comparison: ComparisonType = ComparisonType.GTE
    epsilon: float = 0.01  # used only for eq


@dataclass
class _TestExample:
    input: AnswerClassifierInput
    expectations: List[_TestExpectation] = field(default_factory=list)


@dataclass
class _TestExpectationResult:
    expected: _TestExpectation
    observed: ExpectationClassifierResult
    errors: List[str] = field(default_factory=list)

    def is_failure(self) -> bool:
        return bool(self.errors)


@dataclass
class _TestExampleResult:
    expected: _TestExample
    observed: AnswerClassifierResult
    expectations: List[_TestExpectationResult] = field(default_factory=list)

    def is_failure(self) -> bool:
        return any(x.is_failure() for x in self.expectations)

    def errors(self) -> str:
        if not self.is_failure():
            return ""
        msg = f"Errors for input '{self.expected.input.input_sentence}':"
        for i, x in enumerate(self.expectations):
            for err in x.errors:
                msg += f"\n\t[ex{i if x.expected.expectation == -1 else x.expected.expectation}]: {err}"
        return msg

    def expectations_and_errors(self) -> Tuple[int, int]:
        return len(self.expectations), sum(x.is_failure() for x in self.expectations)


@dataclass
class _TestSet:
    examples: List[_TestExample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _TestSetResultMetrics:
    accuracy: float
    n_expectations: int


@dataclass
class _TestSetResult:
    testset: _TestSet
    results: List[_TestExampleResult] = field(default_factory=list)

    def metrics(self) -> _TestSetResultMetrics:
        n_expectations = 0
        n_errors = 0
        for ex in self.results:
            x, e = ex.expectations_and_errors()
            n_expectations += x
            n_errors += e
        return _TestSetResultMetrics(
            accuracy=1.0 - (n_errors / float(n_expectations))
            if n_expectations > 0
            else 0.0,
            n_expectations=n_expectations,
        )
