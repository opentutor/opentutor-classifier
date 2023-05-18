import json, base64, os
from serverless_modules.constants import ARCH_DEFAULT, MODEL_FILE_NAME
from serverless_modules.utils import create_json_response, require_env
from serverless_modules.logger import get_logger
import boto3

logger = get_logger("check_model")
s3 = boto3.client("s3")
MODELS_BUCKET = require_env("MODELS_BUCKET")
logger.info(f"bucket: {MODELS_BUCKET}")


def handler(event, context):
    try:
        print(json.dumps(event))
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
    except Exception as e:
        logger.error(e)
        return create_json_response(200, e, event)
