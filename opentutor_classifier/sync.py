#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import requests
from pathlib import Path


def sync(lesson: str, url: str, output: str):
    Path(output).mkdir(parents=True, exist_ok=True)
    if url.startswith("http"):
        query = f'query {{ trainingData(lessonId: "{lesson}") {{ config training }} }}'
        request = requests.post(url, json={"query": query})
        training = request.json()["data"]["trainingData"]["training"]
        config = request.json()["data"]["trainingData"]["config"]
    else:
        with open(url) as file:
            data = json.load(file)
            training = data["data"]["trainingData"]["training"]
            config = data["data"]["trainingData"]["config"]
    with open(f"{output}/training.csv", "w+", newline="") as file:
        file.write(training)
    with open(f"{output}/config.yaml", "w+", newline="") as file:
        file.write(config)
