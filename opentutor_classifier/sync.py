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
