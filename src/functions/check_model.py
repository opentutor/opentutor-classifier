#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import base64
import os
from serverless_modules.constants import ARCH_DEFAULT, MODEL_FILE_NAME
from serverless_modules.utils import create_json_response, require_env
from serverless_modules.logger import get_logger
import boto3

logger = get_logger("check_model")
s3 = boto3.client("s3")
MODELS_BUCKET = require_env("MODELS_BUCKET")
logger.info(f"bucket: {MODELS_BUCKET}")


def handler(event, context):
    if "body" not in event:
        raise Exception("bad request: body not in event")
    if event["isBase64Encoded"]:
        body = base64.b64decode(event["body"])
    else:
        body = event["body"]
    payload = json.loads(body)

    if "lesson" not in payload:
        raise Exception("required param: lesson")

    arch = os.environ.get("CLASSIFIER_ARCH") or ARCH_DEFAULT
    lesson = payload["lesson"]
    try:
        model_s3_path = os.path.join(lesson, arch, MODEL_FILE_NAME)
        s3.get_object(**{"Bucket": MODELS_BUCKET, "Key": model_s3_path})
        result = True
    except Exception:
        logger.info(
            f"No model found in memory nor in s3 path {model_s3_path} for lesson: {lesson}"
        )
        result = False
    return create_json_response(200, {"exists": result}, event)
