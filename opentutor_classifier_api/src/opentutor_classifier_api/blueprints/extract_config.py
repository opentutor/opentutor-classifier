#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
import re
from flask import Blueprint, jsonify, request

from cerberus import Validator

from opentutor_classifier import ClassifierConfig
import opentutor_classifier.dao

from opentutor_classifier.classifier_dao import ClassifierDao

extract_config_blueprint = Blueprint("extract_config", __name__)
under_pat = re.compile(r"_([a-z])")


def underscore_to_camel(name: str) -> str:
    return under_pat.sub(lambda x: x.group(1).upper(), name)


def to_camelcase(d: dict) -> dict:
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[underscore_to_camel(k)] = to_camelcase(v)
        elif isinstance(v, list):
            new_d[underscore_to_camel(k)] = [to_camelcase(x) for x in v]
        else:
            new_d[underscore_to_camel(k)] = v
    return new_d


_dao: ClassifierDao = None


def _get_dao() -> ClassifierDao:
    global _dao
    if _dao:
        return _dao
    _dao = ClassifierDao()
    return _dao


@extract_config_blueprint.route("/", methods=["POST"])
@extract_config_blueprint.route("", methods=["POST"])
def evaluate():
    validator = Validator(
        {
            "lesson": {"required": True, "type": "string"},
            "embedding": {"required": False, "type": "boolean", "default": True},
        },
        allow_unknown=True,
        purge_unknown=True,
    )
    if not validator(request.json or {}):
        return jsonify(validator.errors), 400
    args = validator.document
    model_name = args.get("lesson")
    model_roots = [
        os.environ.get("MODEL_ROOT") or "models",
        os.environ.get("MODEL_DEPLOYED_ROOT") or "models_deployed",
    ]
    lesson = args.get("lesson")
    embedding = args.get("embedding", True)
    shared_root = os.environ.get("SHARED_ROOT") or "shared"
    classifier = _get_dao().find_classifier(
        lesson,
        ClassifierConfig(
            dao=opentutor_classifier.dao.find_data_dao(),
            model_name=model_name,
            model_roots=model_roots,
            shared_root=shared_root,
        ),
    )
    config = classifier.save_config_and_model(embedding)
    return (
        jsonify({"output": config}),
        200,
    )
