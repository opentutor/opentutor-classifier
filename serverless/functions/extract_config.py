#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import base64
import os
from serverless.src.utils import create_json_response, require_env

from opentutor_classifier.dao import (
    find_data_dao,
    _CONFIG_YAML,
    MODEL_ROOT_DEFAULT,
    MODELS_DEPLOYED_ROOT_DEFAULT,
)
from opentutor_classifier import ARCH_DEFAULT, ClassifierConfig
from opentutor_classifier.lr2 import MODEL_FILE_NAME
from opentutor_classifier.classifier_dao import ClassifierDao

from serverless.src.logger import get_logger
import boto3

logger = get_logger("extract_config")
s3 = boto3.client("s3")
MODELS_BUCKET = require_env("MODELS_BUCKET")
logger.info(f"bucket: {MODELS_BUCKET}")

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
        return create_json_response(
            400, {"error": "bad request: body not in event"}, event
        )

    if event["isBase64Encoded"]:
        body = base64.b64decode(event["body"])
    else:
        body = event["body"]
    extract_config_body = json.loads(body)

    if "lesson" not in extract_config_body:
        return create_json_response(400, {"error": "lesson is a required param"}, event)
    lesson = extract_config_body["lesson"]
    embedding = (
        extract_config_body["embedding"] if "embedding" in extract_config_body else True
    )
    arch = os.environ.get("CLASSIFIER_ARCH") or ARCH_DEFAULT

    # First confirm that the model and config exists in s3
    try:
        model_s3_path = os.path.join(lesson, arch, MODEL_FILE_NAME)
        config_s3_path = os.path.join(lesson, arch, _CONFIG_YAML)
        logger.info(f"model s3 path: {model_s3_path}")
        logger.info(f"config s3 path: {model_s3_path}")
        s3.head_object(**{"Bucket": MODELS_BUCKET, "Key": model_s3_path})
        s3.head_object(**{"Bucket": MODELS_BUCKET, "Key": config_s3_path})
    except Exception as e:
        logger.error("model or config do not exist in s3, aborting")
        logger.error(e)
        return create_json_response(
            400, {"error": "model or config do not exist in s3, aborting"}, event
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
