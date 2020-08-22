#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
import pickle
from typing import Dict
import yaml

from sklearn import svm

from .dtos import InstanceConfig, InstanceModels


def load_instances(
    model_root="./models",
    models_by_expectation_num_filename="models_by_expectation_num.pkl",
    config_filename="config.yaml",
) -> InstanceModels:
    try:
        with open(path.join(model_root, config_filename)) as config_file:
            config = InstanceConfig(**yaml.load(config_file, Loader=yaml.FullLoader))
    except Exception:
        config = InstanceConfig(question="", expectation_features=[])
    try:
        with open(
            path.join(model_root, models_by_expectation_num_filename), "rb"
        ) as models_file:
            models_by_expectation_num: Dict[int, svm.SVC] = pickle.load(models_file)
    except Exception:
        models_by_expectation_num = {}
    return InstanceModels(
        config=config, models_by_expectation_num=models_by_expectation_num
    )
