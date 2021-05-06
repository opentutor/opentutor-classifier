#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
import pickle
from typing import Dict, Iterable

from sklearn import linear_model

import opentutor_classifier
from opentutor_classifier.utils import find_model_dir
from .dtos import InstanceModels


def load_models(
    model_name: str,
    model_roots: Iterable[str] = ["./models", "./models_deployed"],
    models_by_expectation_num_filename="models_by_expectation_num.pkl",
    # config_filename="config.yaml",
) -> InstanceModels:
    from opentutor_classifier.log import logger

    config = opentutor_classifier.find_data_dao().find_config(model_name)

    logger.warning(
        f"\n\n\n\n\nfind_model_dir for model_name={model_name}, model_roots={model_roots}\n\n\n\n\n"
    )
    model_dir = find_model_dir(model_name, model_roots)
    with open(
        path.join(model_dir, models_by_expectation_num_filename), "rb"
    ) as models_file:
        models_by_expectation_num: Dict[
            int, linear_model.LogisticRegression
        ] = pickle.load(models_file)
        return InstanceModels(
            # config=load_config(path.join(model_dir, config_filename)),
            config=config,
            models_by_expectation_num=models_by_expectation_num,
        )
