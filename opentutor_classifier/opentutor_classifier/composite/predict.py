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
from typing import Dict, Any, Tuple, Union, cast
from opentutor_classifier.log import logger
import time
from async_timeout import timeout


class CompositeAnswerClassifier(AnswerClassifier):

    lr_classifier: AnswerClassifier = LRAnswerClassifier()
    openai_classifier: OpenAIAnswerClassifier = OpenAIAnswerClassifier()

    async def run_lr_evaluate(
        self, answer: AnswerClassifierInput
    ) -> AnswerClassifierResult:
        startTime = time.time()
        logger.info("running lr evaluate " + str(startTime))
        result = self.lr_classifier.evaluate(answer)
        endTime= time.time()
        logger.info("finished lr evaluate " + str(endTime))
        logger.info("lr evaluate took " + str(endTime - startTime))
        return result
    
    async def run_openai_evaluate(
        self, answer: AnswerClassifierInput
    ) -> AnswerClassifierResult:
        try:
            startTime = time.time()
            logger.info("running openai evaluate " + str(startTime))
            result = await self.openai_classifier.evaluate(answer)
            endTime = time.time()
            logger.info("finished openai evaluate " + str(endTime))
            logger.info("openai evaluate took " + str(endTime - startTime))
            return result
        except BaseException as e:
            raise

    def configure(self, config: ClassifierConfig) -> "AnswerClassifier":
        self.lr_classifier = self.lr_classifier.configure(config)
        self.openai_classifier = self.openai_classifier.configure(config)
        return self

    async def evaluate_async(
        self, answer: AnswerClassifierInput
    ) -> AnswerClassifierResult:
        startTime= time.time()
        logger.info("starting composite evaluate " + str(startTime))
        #lr_task = asyncio.wait_for(self.run_lr_evaluate(answer), timeout=20)
        try:
          openai_task = asyncio.wait_for(self.run_openai_evaluate(answer), timeout=1.0)
          await openai_task
        except asyncio.TimeoutError:
            logger.info("openai timed out")
        #results: Tuple[
        #    Union[AnswerClassifierResult, BaseException],
        #    Union[AnswerClassifierResult, BaseException],
        #] = await asyncio.gather(openai_task, return_exceptions=True)
        endTime = time.time()
        logger.info("finished composite evaluate " + str(endTime))

        if type(results[1]) == BaseException:
            return cast(AnswerClassifierResult, results[0])
        else:
            return cast(AnswerClassifierResult, results[1])

    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        return asyncio.run(self.evaluate_async(answer))

    def get_last_trained_at(self) -> float:
        return self.lr_classifier.get_last_trained_at()

    def save_config_and_model(self) -> Dict[str, Any]:
        return self.lr_classifier.save_config_and_model()
