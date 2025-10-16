#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from dataclasses import dataclass, field
import json
from typing import Dict, Generator, List, Union
from dataclass_wizard import JSONWizard
import openai
import backoff
from tiktoken import encoding_for_model
from opentutor_classifier import ExpectationConfig
from opentutor_classifier.config import LABEL_GOOD
from .train import OpenAIGroundTruth
from .constants import (
    OPENAI_API_KEY,
    OPENAI_DEFAULT_TEMP,
    OPENAI_MODEL_LARGE,
    OPENAI_MODEL_SMALL,
    USER_GROUNDTRUTH,
)
from opentutor_classifier.utils import require_env, validate_json
from opentutor_classifier.log import LOGGER


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
    user_groundtruth: Union[OpenAIGroundTruth, None]

    def mask_concept_uuids(self) -> ConceptMask:
        concept_mask: ConceptMask = ConceptMask()
        for index, concept in enumerate(self.user_concepts):
            concept_mask.uuid_to_numbers[concept.expectation_id] = "concept_" + str(
                index
            )
            concept_mask.numbers_to_uuid["concept_" + str(index)] = (
                concept.expectation_id
            )
            concept.expectation_id = "concept_" + str(index)

        if self.user_groundtruth is not None:
            for key in self.user_groundtruth.training_answers.keys():
                entry = self.user_groundtruth.training_answers[key]
                masked_concepts: Dict[str, str] = {}
                for concept_uuid in entry.concepts.keys():
                    if concept_uuid in concept_mask.uuid_to_numbers.keys():
                        masked_concepts[concept_mask.uuid_to_numbers[concept_uuid]] = (
                            str(entry.concepts[concept_uuid].lower() == LABEL_GOOD)
                        )
                entry.concepts = masked_concepts

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
        if self.user_groundtruth is not None:
            result.append(
                {
                    "role": "user",
                    "content": f"{USER_GROUNDTRUTH}\n{json.dumps(self.user_groundtruth.to_dict())}",
                }
            )
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
    logger=LOGGER,
)
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


async def completions_with_backoff(**kwargs) -> Generator:
    return await openai.ChatCompletion.acreate(**kwargs)


async def openai_create(
    call_data: OpenAICall, llm_model_name=None
) -> OpenAIResultContent:
    concept_mask = call_data.mask_concept_uuids()
    messages = call_data.to_openai_json()
    attempts = 0
    result_valid = False
    temperature = OPENAI_DEFAULT_TEMP

    openai.api_key = require_env(OPENAI_API_KEY)

    # logger is not properly logging to cloudwatch.  using print instead for now
    print(f"Sending messages to openAI: {str(messages)}")
    LOGGER.info("Sending messages to OpenAI: " + str(messages))

    if llm_model_name is None:
        if num_tokens_from_string(str(messages), OPENAI_MODEL_SMALL) >= 4000:
            openai_model = OPENAI_MODEL_LARGE
        else:
            openai_model = OPENAI_MODEL_SMALL
    else:
        openai_model = llm_model_name

    while attempts < 5 and not result_valid:
        attempts += 1
        raw_result = await completions_with_backoff(
            model=openai_model, temperature=temperature, messages=messages
        )
        content = raw_result.choices[0].message.content  # type: ignore

        if validate_json(content, OpenAIResultContent):
            result: OpenAIResultContent = OpenAIResultContent.from_json(content)
            if len(
                result.answers[result.answers.__iter__().__next__()].concepts
            ) == len(call_data.user_concepts):
                result_valid = True
                result.answers[
                    result.answers.__iter__().__next__()
                ].unmask_concept_uuids(concept_mask)
                return result
            else:
                temperature += 0.1
                LOGGER.info(
                    "Invalid JSON returned from OpenAI, increasing temperature to "
                    + str(temperature)
                    + " and trying again."
                )

        else:
            temperature += 0.1
            LOGGER.info(
                "Invalid JSON returned from OpenAI, increasing temperature to "
                + str(temperature)
                + " and trying again."
            )

    raise Exception("Unable to get valid JSON from OpenAI after 5 attempts.")
