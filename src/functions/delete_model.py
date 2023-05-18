import json, base64, os
from serverless_modules.constants import _CONFIG_YAML, ARCH_DEFAULT, MODEL_FILE_NAME
from serverless_modules.utils import create_json_response, require_env
from serverless_modules.logger import get_logger
import boto3
import botocore

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
        config_s3_path = os.path.join(lesson, arch, _CONFIG_YAML)
        # Confirm that the model object exists in s3
        s3.head_object(**{"Bucket": MODELS_BUCKET, "Key": model_s3_path})

        s3.delete_object(**{"Bucket": MODELS_BUCKET, "Key": model_s3_path})
        s3.delete_object(**{"Bucket": MODELS_BUCKET, "Key": config_s3_path})

        return create_json_response(200, {"deleted": True}, event)
    except botocore.exceptions.ClientError as e:
        # TODO: explicitly check if the error response is an object not found response, else raise exception
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
