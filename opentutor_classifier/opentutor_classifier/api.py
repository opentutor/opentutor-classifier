#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from dataclasses import dataclass
from io import StringIO
import json
from opentutor_classifier.utils import load_data
import os
import requests
from typing import Optional, TypedDict
import yaml

import pandas as pd

from opentutor_classifier import (
    DataDao,
    FeaturesSaveRequest,
    QuestionConfig,
    TrainingInput,
    dict_to_question_config,
    load_question_config,
)
from opentutor_classifier.log import logger


def get_graphql_endpoint() -> str:
    return os.environ.get("GRAPHQL_ENDPOINT") or "http://graphql/graphql"


def get_api_key() -> str:
    return os.environ.get("API_SECRET") or ""


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


GQL_QUERY_LESSON_CONFIG = """
query LessonConfig($lessonId: String!) {
    me {
        trainingData(lessonId: $lessonId) {
            config
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


def update_features_gql(req: FeaturesSaveRequest) -> GQLQueryBody:
    return {
        "query": GQL_UPDATE_LESSON_FEATURES,
        "variables": {
            "lessonId": req.lesson,
            "expectations": [e.to_dict() for e in req.expectations],
        },
    }


def update_features(req: FeaturesSaveRequest) -> None:
    res_json = __auth_gql(update_features_gql(req))
    if "errors" in res_json:
        raise Exception(json.dumps(res_json.get("errors")))


def __auth_gql(query: GQLQueryBody, url: str = "") -> dict:
    res: Optional[requests.Response] = None
    try:
        res = requests.post(
            url or get_graphql_endpoint(),
            json=query,
            headers={
                "opentutor-api-req": "true",
                "Authorization": f"bearer {get_api_key()}",
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
    data = tdjson["data"]["me"]["trainingData"]
    return dict_to_question_config(yaml.safe_load(data.get("config") or ""))


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


def fetch_all_training_data(url="") -> TrainingInput:
    tdjson = __fetch_all_training_data(url or get_graphql_endpoint())
    if "errors" in tdjson:
        raise Exception(json.dumps(tdjson.get("errors")))
    data = tdjson["data"]["me"]["allTrainingData"]
    df = pd.read_csv(StringIO(data.get("training") or ""))
    df.sort_values(by=["exp_num"], ignore_index=True, inplace=True)
    return TrainingInput(
        lesson="default",
        config=dict_to_question_config(yaml.safe_load(data.get("config") or "")),
        data=df,
    )


class FileDataDao(DataDao):
    def __init__(self, data_root: str):
        self.data_root = data_root

    def _get_config_file(self, lesson: str) -> str:
        return os.path.join(self.data_root, lesson, "config.yaml")

    def find_config(self, lesson: str) -> QuestionConfig:
        return load_question_config(self._get_config_file(lesson))

    def find_training_input(self, lesson: str) -> TrainingInput:
        return TrainingInput(
            lesson=lesson,
            config=self.find_config(lesson),
            data=load_data(os.path.join(self.data_root, lesson, "training.csv")),
        )

    def save_features(self, req: FeaturesSaveRequest) -> None:
        config_file = self._get_config_file(req.lesson)
        config = load_question_config(config_file)
        for e in req.expectations:
            config.expectations[e.expectation].features = e.features
        config.write_to(config_file)


class GqlDataDao(DataDao):
    def find_config(self, lesson: str) -> QuestionConfig:
        return fetch_config(lesson)

    def find_training_input(self, lesson: str) -> TrainingInput:
        return fetch_training_data(lesson)

    def save_features(self, req: FeaturesSaveRequest) -> None:
        update_features(req)
