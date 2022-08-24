#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from flask import Blueprint, jsonify, request
from cerberus import Validator

from opentutor_classifier import ArchLesson, dao, ModelRef, get_classifier_arch
from opentutor_classifier.lr2.constants import MODEL_FILE_NAME

delete_model_blueprint = Blueprint("delete_model", __name__)


@delete_model_blueprint.route("/", methods=["POST"])
@delete_model_blueprint.route("", methods=["POST"])
def delete_model():
    validator = Validator(
        {
            "lesson": {"required": True, "type": "string"},
        },
        allow_unknown=True,
        purge_unknown=True,
    )
    arch = get_classifier_arch()
    if not validator(request.json or {}):
        return jsonify(validator.errors), 400
    args = validator.document
    lesson = args.get("lesson")

    exists = dao.find_data_dao().trained_model_exists(
        ModelRef(arch=arch, lesson=lesson, filename=MODEL_FILE_NAME)
    )

    if not exists:
        return (
            jsonify({"error": f"model for {lesson} does not exist"}),
            500,
        )

    dao.find_data_dao().remove_trained_model(ArchLesson(arch=arch, lesson=lesson))

    return (jsonify({}), 200)
