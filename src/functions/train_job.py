#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import datetime
import os
import boto3
from serverless_modules.utils import require_env
from serverless_modules.logger import get_logger

from serverless_modules.train_job.dao import find_data_dao, _CONFIG_YAML
from serverless_modules.train_job import ClassifierFactory, TrainingConfig
from serverless_modules.constants import (
    MODEL_ROOT_DEFAULT,
    ARCH_DEFAULT,
    MODEL_FILE_NAME,
    DEFAULT_LESSON_NAME,
)

log = get_logger("train-job")
shared = os.environ.get("SHARED_ROOT")
log.info(f"shared: {shared}")
JOBS_TABLE_NAME = require_env("JOBS_TABLE_NAME")
log.info(f"using table {JOBS_TABLE_NAME}")
MODELS_BUCKET = require_env("MODELS_BUCKET")
log.info(f"bucket: {MODELS_BUCKET}")
s3 = boto3.client("s3")
aws_region = os.environ.get("REGION", "us-east-1")
dynamodb = boto3.resource("dynamodb", region_name=aws_region)
job_table = dynamodb.Table(JOBS_TABLE_NAME)


def handler(event, context):

    for record in event["Records"]:
        request = json.loads(str(record["body"]))
        lesson = request["lesson"]
        should_train_default = request["train_default"]
        lesson_name = DEFAULT_LESSON_NAME if should_train_default else lesson
        # ping = request["ping"] if "ping" in request else False
        update_status(request["id"], "IN_PROGRESS")

        try:
            dao = find_data_dao()
            config = TrainingConfig(shared_root=shared)
            fac = ClassifierFactory()

            if should_train_default:
                data = dao.find_default_training_data()
                training = fac.new_training(config, arch=ARCH_DEFAULT)
                training.train_default(data, dao)
            else:
                data = dao.find_training_input(lesson_name)
                training = fac.new_training(config, arch=ARCH_DEFAULT)
                training.train(data, dao)

            # upload model
            s3.upload_file(
                os.path.join(
                    MODEL_ROOT_DEFAULT,
                    ARCH_DEFAULT,
                    lesson_name,
                    MODEL_FILE_NAME,
                ),
                MODELS_BUCKET,
                os.path.join(lesson_name, ARCH_DEFAULT, MODEL_FILE_NAME),
            )

            # upload model config
            s3.upload_file(
                os.path.join(
                    MODEL_ROOT_DEFAULT, ARCH_DEFAULT, lesson_name, _CONFIG_YAML
                ),
                MODELS_BUCKET,
                os.path.join(lesson_name, ARCH_DEFAULT, _CONFIG_YAML),
            )

            update_status(request["id"], "SUCCESS")
        except Exception as e:
            log.exception(e)
            update_status(request["id"], "FAILURE")


def update_status(id, status):
    job_table.update_item(
        Key={"id": id},
        # status is reserved, workaround according to:
        # https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ExpressionAttributeNames.html
        UpdateExpression="set #status = :s, updated = :u",
        ExpressionAttributeNames={
            "#status": "status",
        },
        ExpressionAttributeValues={
            ":s": status,
            ":u": datetime.datetime.now().isoformat(),
        },
    )
