import json
from opentutor_classifier.opentutor_classifier import get_classifier_arch, dao, ModelRef
from opentutor_classifier.opentutor_classifier.lr2.constants import MODEL_FILE_NAME
from serverless_modules.utils import create_json_response


def handler(event):
    print(json.dumps(event))
    payload_body = event["body"]
    if "lesson" not in payload_body:
        return create_json_response(401, payload_body, event)
    arch = get_classifier_arch()
    lesson = event["body"]["lesson"]
    result = dao.find_data_dao().trained_model_exists(
        ModelRef(arch=arch, lesson=lesson, filename=MODEL_FILE_NAME)
    )
    return create_json_response(200, {"exists": result}, event)
