#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
import requests

from flask import Blueprint, jsonify
from opentutor_classifier.api import get_graphql_endpoint  # type: ignore

healthcheck_blueprint = Blueprint("healthcheck", __name__)

GQL_QUERY_STATUS = """
    query Healthcheck {
        health {
            message
            status
        }
    }
"""


@healthcheck_blueprint.route("", methods=["GET"])
@healthcheck_blueprint.route("/", methods=["GET"])
def healthcheck():

    # Get service statuses
    # Admin
    res_admin = requests.head(os.getenv("HEALTHCHECK_ADMIN", "http://admin"))
    admin_status = res_admin.status_code

    # GraphQL
    endpoint = get_graphql_endpoint()

    res_gql = requests.post(endpoint, json={"query": GQL_QUERY_STATUS})
    graphql_status = res_gql.status_code

    # Home
    res_home = requests.head(os.getenv("HEALTHCHECK_HOME", "http://home"))
    home_status = res_home.status_code

    # Tutor
    res_tutor = requests.head(os.getenv("HEALTHCHECK_TUTOR", "http://tutor"))
    tutor_status = res_tutor.status_code

    # Training
    # Needs to ping
    # training_status = requests.get(os.getenv('HEALTHCHECK_TRAINING', "http://training/ping"))
    training_status = 200

    healthy = (
        admin_status == 200
        and graphql_status == 200
        and home_status == 200
        and tutor_status == 200
        and training_status == 200
    )

    return (
        jsonify(
            {
                "services": {
                    "admin": {"status": admin_status},
                    "graphql": {"status": graphql_status},
                    "home": {"status": home_status},
                    "tutor": {"status": tutor_status},
                    "training": {"status": training_status},
                }
            }
        ),
        200 if healthy else 503,
    )
