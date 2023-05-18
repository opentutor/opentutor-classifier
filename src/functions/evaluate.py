#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import os
import base64
import boto3
from serverless_modules.utils import create_json_response, require_env, to_camelcase
from serverless_modules.logger import get_logger

from serverless_modules.evaluate.classifier_dao import ClassifierDao
from serverless_modules.train_job.dao import find_data_dao
from serverless_modules.train_job import ClassifierConfig, AnswerClassifierInput

log = get_logger("status")
JOBS_TABLE_NAME = require_env("JOBS_TABLE_NAME")
log.info(f"using table {JOBS_TABLE_NAME}")
aws_region = os.environ.get("REGION", "us-east-1")
dynamodb = boto3.resource("dynamodb", region_name=aws_region)
job_table = dynamodb.Table(JOBS_TABLE_NAME)


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
    request_body = json.loads(body)

    if "lesson" not in request_body:
        raise Exception("required param: lesson")
    if "input" not in request_body:
        raise Exception("required param: input")

    lesson = request_body["lesson"]
    input_sentence = request_body["input"]
    exp_num = request_body["expectation"] if "expectation" in request_body else ""

    model_roots = [
        os.environ.get("MODEL_ROOT") or "models",
        os.environ.get("MODEL_DEPLOYED_ROOT") or "models_deployed",
    ]
    shared_root = os.environ.get("SHARED_ROOT") or "shared"
    classifier = _get_dao().find_classifier(
        lesson,
        ClassifierConfig(
            dao=find_data_dao(),
            model_name=lesson,
            model_roots=model_roots,
            shared_root=shared_root,
        ),
    )
    _model_op = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_sentence,
            expectation=exp_num,
        )
    )

    return create_json_response(
        200, {"output": to_camelcase(_model_op.to_dict())}, event
    )
