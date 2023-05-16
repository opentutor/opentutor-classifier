import json, os
import boto3
from serverless_modules.utils import create_json_response, require_env
from serverless_modules.logger import get_logger


log = get_logger("status")
JOBS_TABLE_NAME = require_env("JOBS_TABLE_NAME")
log.info(f"using table {JOBS_TABLE_NAME}")
aws_region = os.environ.get("REGION", "us-east-1")
dynamodb = boto3.resource("dynamodb", region_name=aws_region)
job_table = dynamodb.Table(JOBS_TABLE_NAME)


def handler(event, context):
    log.debug(json.dumps(event))
    status_id = event["pathParameters"]["id"]

    db_item = job_table.get_item(Key={"id": status_id})
    log.debug(db_item)
    if "Item" in db_item:
        item = db_item["Item"]
        status = 200
        data = {
            "id": item["id"],
            "status": item["status"],
            "lesson": item["lesson"],
            # only added after trainjob runs
            **({"updated": item["updated"]} if "updated" in item else {}),
            "statusUrl": f"/train/status/{status_id}",
        }
    else:
        data = {
            "error": "not found",
            "message": f"{status_id} not found",
        }
        status = 400

    return create_json_response(status, data, event)
