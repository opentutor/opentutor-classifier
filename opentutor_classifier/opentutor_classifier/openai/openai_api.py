#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from dataclasses import dataclass, field
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
class ConceptMask(JSONWizard):
    uuid_to_numbers: Dict[str, str] = field(default_factory=dict)
    numbers_to_uuid: Dict[str, str] = field(default_factory=dict)


@dataclass
class OpenAICall(JSONWizard):
    system_assignment: str
    user_concepts: List[ExpectationConfig]
    user_answer: List[str]
    user_template: dict
    user_guardrails: str

    def mask_concept_uuids(self) -> ConceptMask:
        concept_mask: ConceptMask = ConceptMask()
        for index, concept in enumerate(self.user_concepts):
            concept_mask.uuid_to_numbers[concept.expectation_id] = "concept_" + str(
                index
            )
            concept_mask.numbers_to_uuid[
                "concept_" + str(index)
            ] = concept.expectation_id
            concept.expectation_id = "concept_" + str(index)

        return concept_mask

    def to_openai_json(self) -> list:
        result: list = []
        result.append({"role": "system", "content": self.system_assignment})
        user_concepts: dict = {}
        for index, concept in enumerate(self.user_concepts):
            user_concepts[concept.expectation_id] = concept.ideal
        result.append({"role": "user", "content": json.dumps(user_concepts)})
        user_answer: dict = {}
        for index, answer in enumerate(self.user_answer):
            user_answer["Answer " + str(index)] = {"Answer Text": answer}
        result.append({"role": "user", "content": json.dumps(user_answer)})
        result.append({"role": "user", "content": json.dumps(self.user_template)})
        result.append({"role": "user", "content": self.user_guardrails})
        return result


@dataclass
class Concept(JSONWizard):
    is_known: bool
    confidence: float
    justification: str


@dataclass
class Answer(JSONWizard):
    answer_text: str
    concepts: Dict[str, Concept]

    def unmask_concept_uuids(self, concept_mask: ConceptMask):
        new_concepts: Dict[str, Concept] = {}
        for key in self.concepts.keys():
            current_concept = self.concepts[key]
            new_concepts[concept_mask.numbers_to_uuid[key]] = current_concept
        self.concepts = new_concepts


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
async def completions_with_backoff(**kwargs) -> Generator:
    return await openai.ChatCompletion.acreate(**kwargs)


async def openai_create(call_data: OpenAICall) -> OpenAIResultContent:
    concept_mask = call_data.mask_concept_uuids()
    messages = call_data.to_openai_json()
    attempts = 0
    result_valid = False
    temperature = OPENAI_DEFAULT_TEMP

    logger.info("Sending messages to OpenAI: " + str(messages))

    while attempts < 5 and not result_valid:
        attempts += 1
        raw_result = await completions_with_backoff(
            model=OPENAI_MODEL, temperature=temperature, messages=messages
        )
        content = raw_result.choices[0].message.content

        if validate_json(content, OpenAIResultContent):
            result_valid = True
            result: OpenAIResultContent = OpenAIResultContent.from_json(content)
            result.answers[result.answers.__iter__().__next__()].unmask_concept_uuids(
                concept_mask
            )
            return result
        else:
            temperature += 0.1
            logger.info(
                "Invalid JSON returned from OpenAI, increasing temperature to "
                + str(temperature)
                + " and trying again."
            )

    raise Exception("Unable to get valid JSON from OpenAI after 5 attempts.")
