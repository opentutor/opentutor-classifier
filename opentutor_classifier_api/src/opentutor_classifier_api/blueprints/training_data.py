#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from opentutor_classifier.api import fetch_training_data
from flask import Blueprint, make_response, request


trainingdata_blueprint = Blueprint("trainingdata", __name__)


@trainingdata_blueprint.route("/<lesson_id>", methods=["GET"])
def get_data(lesson_id: str):
    auth_headers = {"Authorization": request.headers.get("Authorization")}
    data = fetch_training_data(lesson_id, auth_headers=auth_headers)
    data_csv = data.data.to_csv(index=False)
    output = make_response(data_csv)
    output.headers["Content-Disposition"] = "attachment; filename=mentor.csv"
    output.headers["Content-type"] = "text/csv"
    return (
        output,
        200,
    )
