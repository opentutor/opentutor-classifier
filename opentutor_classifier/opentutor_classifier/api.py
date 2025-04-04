#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from dataclasses import dataclass
from datetime import datetime
from dateutil import parser
from io import StringIO
import json
import os
import requests
from typing import Optional, TypedDict, Dict
import yaml


import pandas as pd

from opentutor_classifier import (
    ExpectationFeatures,
    QuestionConfig,
    QuestionConfigSaveReq,
    TrainingInput,
    dict_to_question_config,
)
from opentutor_classifier.log import logger


def get_graphql_endpoint() -> str:
    return os.environ.get("GRAPHQL_ENDPOINT") or "http://graphql/graphql"


def get_sbert_endpoint() -> str:
    return os.environ.get("SBERT_ENDPOINT") or "https://sbert-dev.mentorpal.org/"


def get_api_key() -> str:
    return os.environ.get("API_SECRET") or ""


def get_api_secret_header_and_value() -> tuple[str, str]:
    return (
        os.environ.get("API_WAF_SECRET_HEADER") or "secret-header",
        os.environ.get("API_WAF_SECRET_HEADER_VALUE") or "secret-value",
    )


def get_sbert_api_key() -> str:
    return os.environ.get("SBERT_API_SECRET") or ""


@dataclass
class AnswerUpdateRequest:
    mentor: str
    question: str
    transcript: str


@dataclass
class AnswerUpdateResponse:
    mentor: str
    question: str
    transcript: str


class GQLQueryBody(TypedDict):
    query: str
    variables: dict


GQL_QUERY_LESSON_UPDATED_AT = """
query LessonUpdatedAt($lessonId: String!) {
    me {
        lesson(lessonId: $lessonId) {
            updatedAt
        }
    }
}
"""

GQL_QUERY_LESSON_CONFIG = """
query LessonConfig($lessonId: String!) {
    me {
        config(lessonId: $lessonId) {
            stringified
        }
    }
}
"""


GQL_QUERY_LESSON_TRAINING_DATA = """
query LessonTrainingData($lessonId: String!) {
    me {
        trainingData(lessonId: $lessonId) {
            config
            training
        }
    }
}
"""


GQL_QUERY_ALL_TRAINING_DATA = """
query {
    me {
        allTrainingData {
            config
            training
        }
    }
}
"""


GQL_UPDATE_LESSON_FEATURES = """
mutation UpdateLessonFeatures(
    $lessonId: String!, $expectations: [UpdateExpectationFeaturesInputType]
) {
     me {
        updateLessonFeatures(
            lessonId: $lessonId,
             expectations: $expectations
        ) {
            lessonId
            expectations {
                expectation
                features
            }
        }
    }
}"""


GQL_UPDATE_LAST_TRAINED_AT = """
mutation UpdateLastTrainedAt(
    $lessonId: String!
) {
     me {
        updateLastTrainedAt(
            lessonId: $lessonId,
        ) {
            lessonId
            lastTrainedAt
        }
    }
}"""


def query_lesson_updated_at(lesson: str) -> GQLQueryBody:
    return {
        "query": GQL_QUERY_LESSON_UPDATED_AT,
        "variables": {
            "lessonId": lesson,
        },
    }


def query_all_training_data_gql() -> GQLQueryBody:
    return {"query": GQL_QUERY_ALL_TRAINING_DATA, "variables": {}}


def query_lesson_config_gql(lesson: str) -> GQLQueryBody:
    return {
        "query": GQL_QUERY_LESSON_CONFIG,
        "variables": {
            "lessonId": lesson,
        },
    }


def query_lesson_training_data_gql(lesson: str) -> GQLQueryBody:
    return {
        "query": GQL_QUERY_LESSON_TRAINING_DATA,
        "variables": {
            "lessonId": lesson,
        },
    }


def update_features_gql(req: QuestionConfigSaveReq) -> GQLQueryBody:
    req.config.escape_features()
    result: GQLQueryBody = {
        "query": GQL_UPDATE_LESSON_FEATURES,
        "variables": {
            "lessonId": req.lesson,
            "expectations": [
                ExpectationFeatures(expectation=i, features=e.features).to_dict()
                for i, e in enumerate(req.config.expectations)
            ],
        },
    }
    req.config.unescape_features()
    return result


def update_last_trained_at_gql(lesson: str) -> GQLQueryBody:
    return {
        "query": GQL_UPDATE_LAST_TRAINED_AT,
        "variables": {
            "lessonId": lesson,
        },
    }


def update_last_trained_at(lesson: str) -> None:
    res_json = __auth_gql(update_last_trained_at_gql(lesson))
    if "errors" in res_json:
        raise Exception(json.dumps(res_json.get("errors")))


def update_features(req: QuestionConfigSaveReq) -> None:
    res_json = __auth_gql(update_features_gql(req))
    if "errors" in res_json:
        raise Exception(json.dumps(res_json.get("errors")))


def __auth_gql(query: GQLQueryBody, url: str = "") -> dict:
    res: Optional[requests.Response] = None
    secret_header, secret_value = get_api_secret_header_and_value()
    try:
        res = requests.post(
            url or get_graphql_endpoint(),
            json=query,
            headers={
                "opentutor-api-req": "true",
                "Authorization": f"bearer {get_api_key()}",
                secret_header: secret_value,
            },
        )
        res.raise_for_status()
        return res.json()
    except BaseException as x:
        logger.warning(f"error for query: {query}")
        if res:
            logger.warning(f"res={res.text}")
        logger.exception(x)
        raise x


def __fetch_training_data(lesson: str, url: str) -> dict:
    if url and not url.startswith("http"):
        with open(url) as f:
            return json.load(f)
    return __auth_gql(query_lesson_training_data_gql(lesson), url=url)


def fetch_config(lesson: str) -> QuestionConfig:
    tdjson = __auth_gql(query_lesson_config_gql(lesson))
    if "errors" in tdjson:
        raise Exception(json.dumps(tdjson.get("errors")))
    data = tdjson["data"]["me"]["config"]
    return dict_to_question_config(yaml.safe_load(data.get("stringified") or ""))


def fetch_lesson_updated_at(lesson: str) -> datetime:
    tdjson = __auth_gql(query_lesson_updated_at(lesson))
    if "errors" in tdjson:
        raise Exception(json.dumps(tdjson.get("errors")))
    updated_at = tdjson["data"]["me"]["lesson"]["updatedAt"]
    # can't use date.fromisoformat because it doesn't handle Z suffix
    return parser.isoparse(updated_at)


def fetch_training_data(lesson: str, url="") -> TrainingInput:
    tdjson = __fetch_training_data(lesson, url or get_graphql_endpoint())
    if "errors" in tdjson:
        raise Exception(json.dumps(tdjson.get("errors")))
    data = tdjson["data"]["me"]["trainingData"]
    df = pd.read_csv(StringIO(data.get("training") or ""))
    df.sort_values(by=["exp_num"], ignore_index=True, inplace=True)
    return TrainingInput(
        lesson=lesson,
        config=dict_to_question_config(yaml.safe_load(data.get("config") or "")),
        data=df,
    )


def __fetch_all_training_data(url: str) -> dict:
    if url and not url.startswith("http"):
        with open(url) as f:
            return json.load(f)
    return __auth_gql(query_all_training_data_gql(), url=url)


def fetch_all_training_data(url="") -> pd.DataFrame:
    tdjson = __fetch_all_training_data(url or get_graphql_endpoint())
    if "errors" in tdjson:
        raise Exception(json.dumps(tdjson.get("errors")))
    data = tdjson["data"]["me"]["allTrainingData"]
    df = pd.read_csv(StringIO(data.get("training") or ""))
    df.sort_values(by=["exp_num"], ignore_index=True, inplace=True)
    return df


def get_sbert_waf_secret_header():
    return os.environ.get("SBERT_WAF_SECRET_HEADER") or ""


def get_sbert_waf_secret_value():
    return os.environ.get("SBERT_WAF_SECRET_VALUE") or ""


def sbert_word_to_vec(words: list, slim: bool = False):
    model_name = "word2vec_slim" if slim else "word2vec"

    # sbert WAF only allows 8kb body size, 800 words is ~6kb
    req_words_chunk_size = 800
    req_words_chunks = [
        words[i : i + req_words_chunk_size]
        for i in range(0, len(words), req_words_chunk_size)
    ]
    final_res: Dict = {}
    logger.info(f"Number of chunk requests to make: {len(req_words_chunks)}")

    for i, req_words in enumerate(req_words_chunks):

        words_appended_by_space = " ".join(req_words)
        res = requests.post(
            f"{get_sbert_endpoint()}v1/w2v",
            headers={
                "Authorization": f"bearer {get_sbert_api_key()}",
                f"{get_sbert_waf_secret_header()}": f"{get_sbert_waf_secret_value()}",
            },
            json={"model": model_name, "words": words_appended_by_space},
        )
        res.raise_for_status()
        logger.info(f"Finished req_chuck #{i+1}")
        final_res = {**final_res, **res.json()}
    return final_res


def get_sbert_index_to_key(slim: bool = False):
    model_name = "word2vec_slim" if slim else "word2vec"
    res = requests.post(
        f"{get_sbert_endpoint()}v1/w2v/index_to_key?model={model_name}",
        headers={
            "Authorization": f"bearer {get_sbert_api_key()}",
            f"{get_sbert_waf_secret_header()}": f"{get_sbert_waf_secret_value()}",
        },
    )
    res.raise_for_status()
    return res.json()
