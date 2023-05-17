import json, datetime, os
import boto3
from serverless_modules.utils import require_env
from serverless_modules.logger import get_logger

from serverless_modules.train_job.dao import find_data_dao
from serverless_modules.train_job import ClassifierFactory, TrainingConfig
from serverless_modules.train_job.constants import (
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
        ping = request["ping"] if "ping" in request else False
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
