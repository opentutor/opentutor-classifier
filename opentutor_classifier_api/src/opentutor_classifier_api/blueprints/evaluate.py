#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
from flask import Blueprint, jsonify, request

from cerberus import Validator

from opentutor_classifier import AnswerClassifierInput
from opentutor_classifier.svm import SVMAnswerClassifier
from opentutor_classifier.svm.utils import dict_to_config, find_model_dir

import json
import re

eval_blueprint = Blueprint("evaluate", __name__)
under_pat = re.compile(r"_([a-z])")


def underscore_to_camel(name: str) -> str:
    return under_pat.sub(lambda x: x.group(1).upper(), name)


def to_camelcase(d: dict) -> dict:
    new_d = {}
    for k, v in d.items():
        new_d[underscore_to_camel(k)] = to_camelcase(v) if isinstance(v, dict) else v
    return new_d


@eval_blueprint.route("/", methods=["POST"])
@eval_blueprint.route("", methods=["POST"])
def evaluate():
    validator = Validator(
        {
            "lesson": {"required": True, "type": "string"},
            "input": {"required": True, "type": "string"},
            "expectation": {
                "required": False,
                "type": "integer",
                "coerce": int,
                "default": -1,
            },
            "config": {
                "required": False,
                "type": "dict",
                "schema": {
                    "question": {"type": "string", "required": False},
                    "expectations": {"type": "list", "required": False},
                },
            },
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
    question_models = find_model_dir(model_name, model_roots=model_roots)
    config_data = {}
    input_sentence = args.get("input")
    exp_num = int(args.get("expectation", -1))
    if not question_models:
        if not args.get("config"):
            return (
                jsonify(
                    {
                        "message": f"No models found for lesson {args.get('lesson')}. Config data is required"
                    }
                ),
                404,
            )
        else:
            model_name = "default"
            question_models = find_model_dir(model_name, model_roots=model_roots)
            config_data = args.get("config")
    version_path = os.path.join(question_models, "build_version.json")
    version = None
    if os.path.isfile(version_path):
        with open(version_path) as f:
            version = json.load(f)
    shared_root = os.environ.get("SHARED_ROOT") or "shared"
    classifier = SVMAnswerClassifier(
        model_name, model_roots=model_roots, shared_root=shared_root
    )
    _model_op = classifier.evaluate(
        AnswerClassifierInput(
            input_sentence=input_sentence,
            config_data=dict_to_config(config_data),
            expectation=exp_num,
        )
    )
    return (
        jsonify({"output": to_camelcase(_model_op.to_dict()), "version": version}),
        200,
    )
