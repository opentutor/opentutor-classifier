#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
from serverless.src.utils import create_json_response
from opentutor_classifier.api import fetch_training_data


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
