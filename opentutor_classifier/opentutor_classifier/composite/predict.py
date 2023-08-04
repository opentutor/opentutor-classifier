#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from opentutor_classifier import (
    AnswerClassifier,
    ClassifierConfig,
    AnswerClassifierInput,
    AnswerClassifierResult,
)
import asyncio
from opentutor_classifier.lr2.predict import LRAnswerClassifier
from opentutor_classifier.openai.predict import OpenAIAnswerClassifier
from typing import Dict, Any, Union


class CompositeAnswerClassifier(AnswerClassifier):

    lr_classifier: LRAnswerClassifier = LRAnswerClassifier()
    openai_classifier: OpenAIAnswerClassifier = OpenAIAnswerClassifier()

    async def run_lr_evaluate(
        self, answer: AnswerClassifierInput
    ) -> AnswerClassifierResult:
        return self.lr_classifier.evaluate(answer)

    async def run_openai_evaluate(
        self, answer: AnswerClassifierInput
    ) -> AnswerClassifierResult:
        return self.openai_classifier.evaluate(answer)

    def configure(self, config: ClassifierConfig) -> "AnswerClassifier":
        self.lr_classifier = self.lr_classifier.configure(config)
        self.openai_classifier = self.openai_classifier.configure(config)

    async def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        lr_task = asyncio.wait_for(self.run_lr_evaluate(answer))
        openai_task = asyncio.wait_for(self.run_openai_evaluate(answer), timeout=10)
        results: list[Union[AnswerClassifierResult, Exception]] = await asyncio.gather(
            lr_task, openai_task, return_exceptions=True
        )
        if type(results[1]) == Exception:
            return results[0]
        else:
            return results[1]

    def get_last_trained_at(self) -> float:
        return self.lr_classifier.get_last_trained_at()

    def save_config_and_model(self) -> Dict[str, Any]:
        return self.lr_classifier.save_config_and_model()
