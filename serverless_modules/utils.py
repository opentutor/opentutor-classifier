#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
import requests
import yaml
import re

from tqdm import tqdm
from os import _Environ, environ, path
import json
from typing import Union, Dict, Any, Iterable
from pathlib import Path


def prop_bool(
    name: str, props: Union[Dict[str, Any], _Environ], dft: bool = False
) -> bool:
    if not (props and name in props):
        return dft
    v = props[name]
    return str(v).lower() in ["1", "t", "true"]


def model_last_updated_at(
    arch: str, model_name: str, model_roots: Iterable[str], model_file_name: str
) -> float:
    dir_path = find_model_dir(arch, model_name, model_roots)
    file_path = path.join(dir_path, model_file_name)
    return Path(file_path).stat().st_mtime if path.isfile(file_path) else -1.0


def find_model_dir(arch: str, model_name: str, model_roots: Iterable[str]) -> str:
    for m in model_roots:
        d = path.join(m, arch, model_name)
        if path.isdir(d):
            return d
    return ""


def download(url: str, to_path: str):
    print("downloading")
    print(f"\tfrom {url}")
    print(f"\tto {to_path}")
    r = requests.get(url, stream=True)
    total_size_in_bytes = int(r.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    os.makedirs(os.path.dirname(to_path), exist_ok=True)
    with open(to_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=block_size):
            if chunk:
                progress_bar.update(len(chunk))
                f.write(chunk)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception("failed to fully download")


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
    headers[
        "Access-Control-Allow-Methods"
    ] = "Authorization,Origin,Accept,Accept-Language,Content-Language,Content-Type"


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)


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
