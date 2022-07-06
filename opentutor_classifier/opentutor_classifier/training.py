#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path

import opentutor_classifier
from . import (
    ClassifierFactory,
    TrainingConfig,
    TrainingResult,
)
from . import DataDao
from .dao import FileDataDao


def train(
    lesson: str,
    arch="",
    config: TrainingConfig = None,
    dao: DataDao = None,
) -> TrainingResult:
    dao = dao or opentutor_classifier.dao.find_data_dao()
    data = dao.find_training_input(lesson)
    fac = ClassifierFactory()
    training = fac.new_training(config or TrainingConfig(), arch=arch)
    res = training.train(data, dao)
    return res


def train_data_root(
    arch="", config: TrainingConfig = None, data_root="data", output_dir=""
) -> TrainingResult:
    droot, lesson = path.split(path.abspath(data_root))
    return train(
        lesson,
        arch=arch,
        config=config,
        dao=FileDataDao(droot, model_root=output_dir),
    )


def train_default_data_root(
    arch="", config: TrainingConfig = None, data_root="data", output_dir=""
) -> TrainingResult:
    droot, __default__ = path.split(path.abspath(data_root))
    return train_default(
        arch=arch, config=config, dao=FileDataDao(droot, model_root=output_dir)
    )


def train_default(
    arch="",
    config: TrainingConfig = None,
    dao: DataDao = None,
) -> TrainingResult:
    dao = dao or opentutor_classifier.dao.find_data_dao()
    data = dao.find_default_training_data()
    return (
        ClassifierFactory()
        .new_training(config or TrainingConfig(), arch=arch)
        .train_default(data, dao)
    )
