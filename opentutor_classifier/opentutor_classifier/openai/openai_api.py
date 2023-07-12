#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from dataclasses import dataclass
import json
from typing import Dict, Generator, List
from dataclass_wizard import JSONWizard
import openai
import backoff
from opentutor_classifier import ExpectationConfig
from .constants import OPENAI_API_KEY, OPENAI_DEFAULT_TEMP, OPENAI_MODEL
from opentutor_classifier.utils import require_env, validate_json
from opentutor_classifier.log import logger

openai.api_key = require_env(OPENAI_API_KEY)


@dataclass
class OpenAICall(JSONWizard):
    system_assignment: str
    user_concepts: List[ExpectationConfig]
    user_answer: List[str]
    user_template: dict
    user_guardrails: str

    def to_openai_json(self) -> str:
        result: dict = {}
        result["system-assignment"] = self.system_assignment
        user_concepts: dict = {}
        for index, concept in enumerate(self.user_concepts):
            user_concepts[concept.expectation_id] = concept.ideal
        result["user-concepts"] = user_concepts
        user_answer: dict = {}
        for index, answer in enumerate(self.user_answer):
            user_answer["Answer " + str(index)] = {"Answer Text": answer}
        result["user-answer"] = user_answer
        result["user-template"] = self.user_template
        result["user-guardrails"] = self.user_guardrails
        return json.dumps(result, indent=2)


@dataclass
class Concept(JSONWizard):
    is_known: bool
    confidence: float
    justification: str


@dataclass
class Answer(JSONWizard):
    answer_text: str
    concepts: Dict[str, Concept]


@dataclass
class OpenAIResultContent(JSONWizard):
    answers: Dict[str, Answer]


@backoff.on_exception(
    backoff.expo,
    (
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.Timeout,
    ),
    logger=logger,
)
def completions_with_backoff(**kwargs) -> Generator:
    return openai.ChatCompletion.create(**kwargs)


def openai_create(call_data: OpenAICall) -> OpenAIResultContent:
    mesasges = {"role": "user", "content": call_data.to_openai_json()}

    attempts = 0
    result_valid = False
    temperature = OPENAI_DEFAULT_TEMP

    while attempts < 5 and not result_valid:
        attempts += 1
        raw_result = completions_with_backoff(
            model=OPENAI_MODEL, temperature=temperature, messages=mesasges
        )
        content = raw_result.choices[0].message.content

        if validate_json(content, OpenAIResultContent):
            result_valid = True
            result: OpenAIResultContent = OpenAIResultContent.from_json(content)
            return result
        else:
            temperature += 0.1
            logger.info(
                "Invalid JSON returned from OpenAI, increasing temperature to "
                + str(temperature)
                + " and trying again."
            )

    raise Exception("Unable to get valid JSON from OpenAI after 5 attempts.")
