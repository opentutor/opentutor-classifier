#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import base64
import os
from serverless.src.constants import ARCH_DEFAULT, MODEL_FILE_NAME, _CONFIG_YAML
from serverless.src.utils import create_json_response, require_env
from serverless.src.logger import get_logger
import boto3
import botocore

logger = get_logger("check_model")
s3 = boto3.client("s3")
MODELS_BUCKET = require_env("MODELS_BUCKET")
logger.info(f"bucket: {MODELS_BUCKET}")


def handler(event, context):
    if "body" not in event:
        return create_json_response(
            400, {"error": "bad request: body not in event"}, event
        )

    if event["isBase64Encoded"]:
        body = base64.b64decode(event["body"])
    else:
        body = event["body"]
    payload = json.loads(body)

    if "lesson" not in payload:
        return create_json_response(400, {"error": "lesson is a required param"}, event)

    arch = os.environ.get("CLASSIFIER_ARCH") or ARCH_DEFAULT
    lesson = payload["lesson"]

    try:
        model_s3_path = os.path.join(lesson, arch, MODEL_FILE_NAME)
        config_s3_path = os.path.join(lesson, arch, _CONFIG_YAML)
        # Confirm that the model object exists in s3
        s3.head_object(**{"Bucket": MODELS_BUCKET, "Key": model_s3_path})

        s3.delete_object(**{"Bucket": MODELS_BUCKET, "Key": model_s3_path})
        s3.delete_object(**{"Bucket": MODELS_BUCKET, "Key": config_s3_path})

        return create_json_response(200, {"deleted": True}, event)
    except botocore.exceptions.ClientError as e:
        # if not a 404, then an unexpected error occured
        if e.response["Error"]["Code"] != "404":
            logger.error(e)
            raise e
        logger.info(f"No model found in s3 at path {model_s3_path}")
        return create_json_response(
            200,
            {
                "deleted": False,
                "reason": f"No model found in s3 at path {model_s3_path}",
            },
            event,
        )
