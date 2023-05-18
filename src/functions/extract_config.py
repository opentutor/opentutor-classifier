import json, base64, os
from serverless_modules.utils import create_json_response
from serverless_modules.train_job.constants import (
    MODEL_ROOT_DEFAULT,
    MODELS_DEPLOYED_ROOT_DEFAULT,
)
from serverless_modules.evaluate.classifier_dao import ClassifierDao, ClassifierConfig
from serverless_modules.train_job.dao import find_data_dao


shared_root = os.environ.get("SHARED_ROOT") or "shared"
model_roots = [
    os.environ.get("MODEL_ROOT") or MODEL_ROOT_DEFAULT,
    os.environ.get("MODEL_DEPLOYED_ROOT") or MODELS_DEPLOYED_ROOT_DEFAULT,
]

_dao: ClassifierDao = None


def _get_dao() -> ClassifierDao:
    global _dao
    if _dao:
        return _dao
    _dao = ClassifierDao()
    return _dao


def handler(event, context):
    print(json.dumps(event))

    if "body" not in event:
        raise Exception("bad request: body not in event")

    if event["isBase64Encoded"]:
        body = base64.b64decode(event["body"])
    else:
        body = event["body"]
    extract_config_body = json.loads(body)

    if "lesson" not in extract_config_body:
        raise Exception("required param: lesson")
    lesson = extract_config_body["lesson"]
    embedding = (
        extract_config_body["embedding"] if "embedding" in extract_config_body else True
    )

    classifier = _get_dao().find_classifier(
        lesson,
        ClassifierConfig(
            dao=find_data_dao(),
            model_name=lesson,
            model_roots=model_roots,
            shared_root=shared_root,
        ),
    )
    config = classifier.save_config_and_model(embedding)

    return create_json_response(200, {"output": config}, event)
