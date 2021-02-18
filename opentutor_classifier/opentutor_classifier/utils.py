#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
from typing import Any, Dict, Iterable, Optional
import pandas as pd
import yaml

from . import ExpectationConfig, QuestionConfig


# TODO this should never return None, but code currently depends on that
def dict_to_config(config_data: dict) -> Optional[QuestionConfig]:
    return (
        QuestionConfig(
            question=config_data.get("question", ""),
            expectations=[
                ExpectationConfig(ideal=i["ideal"])
                for i in config_data.get("expectations", [])
            ],
        )
        if config_data
        else None
    )


def find_model_dir(model_name: str, model_roots: Iterable[str]) -> str:
    for m in model_roots:
        d = path.join(m, model_name)
        if path.isdir(d):
            return d
    return ""


def load_config(config_file: str) -> QuestionConfig:
    with open(config_file) as f:
        return QuestionConfig(**yaml.load(f, Loader=yaml.FullLoader))


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, encoding="latin-1")


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)
