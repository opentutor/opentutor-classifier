#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from typing import Generator
import openai
import backoff
from .constants import OPENAI_API_KEY, OPENAI_DEFAULT_TEMP, OPENAI_MODEL
from .__init__ import OpenAICall, OpenAIResultContent
from utils import require_env, validate_json
from log import logger

openai.api_key = require_env(OPENAI_API_KEY)


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
def completions_with_backoff(self, **kwargs) -> Generator:
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
        content = raw_result["content"]

        if validate_json(content, OpenAIResultContent):
            result_valid = True
            result = OpenAIResultContent.from_json(content)
            return result
        else:
            temperature += 0.1
            logger.info(
                "Invalid JSON returned from OpenAI, increasing temperature to "
                + str(temperature)
                + " and trying again."
            )

    raise Exception(
        "Invalid JSON returned from OpenAI after 5 attempts, returning None."
    )
