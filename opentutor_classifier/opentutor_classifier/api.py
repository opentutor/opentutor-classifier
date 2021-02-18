#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from io import StringIO
import json
import os
import requests
import yaml

import pandas as pd

from opentutor_classifier import TrainingInput, dict_to_question_config

GRAPHQL_ENDPOINT = os.environ.get("GRAPHQL_ENDPOINT") or "http://graphql/graphql"
API_SECRET = os.environ.get("API_SECRET") or ""


def __fetch_training_data(lesson: str, url: str) -> dict:
    if not url.startswith("http"):
        with open(url) as f:
            return json.load(f)
    res = requests.post(
        url,
        json={
            "query": f'query {{ me {{ trainingData(lessonId: "{lesson}") {{ config training }} }} }}'
        },
        headers={'"opentutor-api-req': "true", "Authorization": "bearer {API_SECRET}"},
    )
    res.raise_for_status()
    return res.json()


def fetch_training_data(lesson: str, url=GRAPHQL_ENDPOINT) -> TrainingInput:
    tdjson = __fetch_training_data(lesson, url or GRAPHQL_ENDPOINT)
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
