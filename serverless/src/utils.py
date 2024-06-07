#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import json
import re
from os import environ


def append_secure_headers(headers):
    secure = {
        "content-security-policy": "upgrade-insecure-requests;",
        "referrer-policy": "no-referrer-when-downgrade",
        "strict-transport-security": "max-age=31536000",
        "x-content-type-options": "nosniff",
        "x-frame-options": "SAMEORIGIN",
        "x-xss-protection": "1; mode=block",
    }
    for h in secure:
        headers[h] = secure[h]


def append_cors_headers(headers, event):
    origin = environ.get("CORS_ORIGIN", "*")
    # TODO specify allowed list of origins and if event["headers"]["origin"] is one of them then allow it
    # if "origin" in event["headers"] and getenv.array('CORS_ORIGIN').includes(event["headers"]["origin"]):
    #     origin = event["headers"]["origin"]

    headers["Access-Control-Allow-Origin"] = origin
    headers["Access-Control-Allow-Origin"] = "*"
    headers["Access-Control-Allow-Headers"] = "GET,PUT,POST,DELETE,OPTIONS"
    headers["Access-Control-Allow-Methods"] = (
        "Authorization,Origin,Accept,Accept-Language,Content-Language,Content-Type"
    )


def create_json_response(status, data, event, headers={}):
    body = {"data": data}
    append_cors_headers(headers, event)
    append_secure_headers(headers)
    response = {"statusCode": status, "body": json.dumps(body), "headers": headers}
    return response


def require_env(n: str) -> str:
    env_val = environ.get(n, "")
    if not env_val:
        raise EnvironmentError(f"missing required env var {n}")
    return env_val


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
