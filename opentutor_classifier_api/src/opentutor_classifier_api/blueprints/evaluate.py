#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from dataclasses import dataclass
from dataclass_wizard import JSONWizard
import os
import re
import asyncio
from flask import Blueprint, jsonify, request
from typing import Any
from cerberus import Validator

from opentutor_classifier import (
    AnswerClassifierInput,
    AnswerClassifierResult,
    ClassifierConfig,
    ARCH_DEFAULT,
)
import opentutor_classifier.dao

from opentutor_classifier.classifier_dao import ClassifierDao

eval_blueprint = Blueprint("evaluate", __name__)
under_pat = re.compile(r"_([a-z])")


@dataclass
class Output(JSONWizard):
    output: AnswerClassifierResult


def underscore_to_camel(name: str) -> str:
    return under_pat.sub(lambda x: x.group(1).upper(), name)


def to_camelcase(d: Any) -> dict:
    if isinstance(d, list):
        new_d = []
        for x in d:
            new_d.append(to_camelcase(x))
        return new_d
    elif isinstance(d, dict):
        for k, v in d.items():
            new_d = {}
            if isinstance(v, dict):
                new_d[underscore_to_camel(k)] = to_camelcase(v)
            elif isinstance(v, list):
                new_d[underscore_to_camel(k)] = [to_camelcase(x) for x in v]
            else:
                new_d[underscore_to_camel(k)] = v
        return new_d
    else:
        return d


_dao: ClassifierDao = None


def _get_dao() -> ClassifierDao:
    global _dao
    if _dao:
        return _dao
    _dao = ClassifierDao()
    return _dao


@eval_blueprint.route("/", methods=["POST"])
@eval_blueprint.route("", methods=["POST"])
def evaluate():
    validator = Validator(
        {
            "lesson": {"required": True, "type": "string"},
            "input": {"required": True, "type": "string"},
            "expectation": {
                "required": False,
                "type": "string",
                "coerce": str,
                "default": "",
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
    input_sentence = args.get("input")
    exp_num = args.get("expectation", "")
    lesson = args.get("lesson")
    arch = args.get("arch", ARCH_DEFAULT)
    shared_root = os.environ.get("SHARED_ROOT") or "shared"
    dao = opentutor_classifier.dao.find_data_dao()
    config = dao.find_training_config(lesson)

    classifier = _get_dao().find_classifier(
        lesson,
        ClassifierConfig(
            dao=dao,
            model_name=model_name,
            model_roots=model_roots,
            shared_root=shared_root,
        ),
        arch,
    )
    _model_op = asyncio.run(
        classifier.evaluate(
            AnswerClassifierInput(
                input_sentence=input_sentence,
                config_data=config,
                expectation=exp_num,
            )
        )
    )
    return (
        jsonify(Output(output=_model_op).to_json()),
        200,
    )
