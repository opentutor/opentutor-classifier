#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import environ
from flask import Blueprint, jsonify, request

import opentutor_classifier_tasks
import opentutor_classifier_tasks.tasks


train_default_blueprint = Blueprint("train_default", __name__)


def _to_status_url(root: str, id: str) -> str:
    return f"{request.url_root.replace('http://', 'https://', 1) if (environ.get('STATUS_URL_FORCE_HTTPS') or '').lower() in ('1', 'y', 'true', 'on') and str.startswith(request.url_root,'http://') else request.url_root}classifier/train_default/status/{id}"


@train_default_blueprint.route("/", methods=["POST"])
@train_default_blueprint.route("", methods=["POST"])
def train():
    t = opentutor_classifier_tasks.tasks.train_default_task.apply_async(args=[])
    return jsonify(
        {
            "data": {
                "id": t.id,
                "statusUrl": _to_status_url(request.url_root, t.id),
            }
        }
    )


@train_default_blueprint.route("/status/<task_id>/", methods=["GET"])
@train_default_blueprint.route("/status/<task_id>", methods=["GET"])
def train_status(task_id: str):
    t = opentutor_classifier_tasks.tasks.train_default_task.AsyncResult(task_id)
    return jsonify(
        {
            "data": {
                "id": task_id,
                "state": t.state or "PENDING",
                "status": t.status,
                "info": None
                if not t.info
                else t.info
                if isinstance(t.info, dict) or isinstance(t.info, list)
                else str(t.info),
            }
        }
    )
