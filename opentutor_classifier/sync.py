import csv
import json
import requests
from pathlib import Path


def sync(lesson_id: str, api_url: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sync_training(lesson_id, api_url, output_dir)
    sync_config(lesson_id, api_url, output_dir)


def sync_training(lesson_id: str, api_url: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    query = f'query {{ lessonTrainingData(lessonId: "{lesson_id}") }}'
    request = requests.post(api_url, json={"query": query})
    lessonTrainingData = json.loads(request.text)["data"]["lessonTrainingData"]

    file = open(f"{output_dir}/training.csv", "w+", newline="")
    writer = csv.writer(file)
    for line in lessonTrainingData.split("\n"):
        csv_data = line.split(",")
        writer.writerow([csv_data[0], csv_data[1], csv_data[2]])
    print(file.read())


def sync_config(lesson_id: str, api_url: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    query = f'query {{ lesson(lessonId: "{lesson_id}") {{ question }} }}'
    request = requests.post(api_url, json={"query": query})
    question = json.loads(request.text)["data"]["lesson"]["question"]

    file = open(f"{output_dir}/config.yaml", "w+", newline="")
    file.write(f'question: "{question}"')
    print(file.read())
    file.close()
