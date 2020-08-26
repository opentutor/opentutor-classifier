#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from io import StringIO
import json
import requests
import yaml

import pandas as pd

from opentutor_classifier import TrainingInput

GRAPHQL_ENDPOINT = "http://graphql"


def __fetch_training_data(lesson: str, url: str) -> dict:
    if not url.startswith("http"):
        with open(url) as f:
            return json.load(f)
    res = requests.post(
        url,
        json={
            "query": f'query {{ trainingData(lessonId: "{lesson}") {{ config training }} }}'
        },
    )
    res.raise_for_status()
    return res.json()


def fetch_training_data(lesson: str, url=GRAPHQL_ENDPOINT) -> TrainingInput:
    # if not url.startswith("http"):
    #     with open(url) as file:
    #         data = json.load(file)
    #         training = data["data"]["trainingData"]["training"]
    #         config = data["data"]["trainingData"]["config"]
    # res = requests.post(
    #     url,
    #     json={
    #         "query": f'query {{ trainingData(lessonId: "{lesson}") {{ config training }} }}'
    #     },
    # )
    # res.raise_for_status()
    tdjson = __fetch_training_data(lesson, url)
    if "errors" in tdjson:
        raise Exception(json.dumps(tdjson.get("errors")))
    data = tdjson["data"]["trainingData"]
    return TrainingInput(
        lesson=lesson,
        config=yaml.safe_load(data.get("config") or ""),
        data=pd.read_csv(StringIO(data.get("training") or "")),
    )
