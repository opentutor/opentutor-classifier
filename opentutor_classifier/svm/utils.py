#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import logging
from os import path
import pickle
from typing import Dict, Optional
import yaml

from sklearn import svm

from opentutor_classifier import ExpectationFeatures, QuestionConfig
from .dtos import InstanceModels


def load_config(config_file: str) -> QuestionConfig:
    logging.warning(f"logging config from {config_file}...")
    with open(config_file) as f:
        return QuestionConfig(**yaml.load(f, Loader=yaml.FullLoader))


def load_instances(
    model_root="./models",
    models_by_expectation_num_filename="models_by_expectation_num.pkl",
    config_filename="config.yaml",
) -> InstanceModels:
    with open(
        path.join(model_root, models_by_expectation_num_filename), "rb"
    ) as models_file:
        models_by_expectation_num: Dict[int, svm.SVC] = pickle.load(models_file)
        return InstanceModels(
            config=load_config(path.join(model_root, config_filename)),
            models_by_expectation_num=models_by_expectation_num,
        )


# TODO this should never return None, but code currently depends on that
def dict_to_config(config_data: dict) -> Optional[QuestionConfig]:
    return (
        QuestionConfig(
            question=config_data.get("question", ""),
            expectations=[
                ExpectationFeatures(ideal=i["ideal"])
                for i in config_data.get("expectations", [])
            ],
        )
        if config_data
        else None
    )
