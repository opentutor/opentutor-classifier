#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import os
import boto3
from src.utils import create_json_response, require_env
from src.logger import get_logger


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
            "state": item["status"],
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
