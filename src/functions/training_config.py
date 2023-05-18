import json
from serverless_modules.utils import create_json_response
from serverless_modules.train_job.api import fetch_training_data


def handler(event, context):
    print(json.dumps(event))
    lesson_id = event["pathParameters"]["lesson_id"]
    data = fetch_training_data(lesson_id)
    data_config = data.config.to_dict()
    extra_headers = {
        "Content-Disposition": "attachment; filename=mentor.csv",
        "Content-type": "text/csv",
    }
    return create_json_response(200, data_config, event, extra_headers)
