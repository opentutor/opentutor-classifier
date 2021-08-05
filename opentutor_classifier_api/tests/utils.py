#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from contextlib import contextmanager
from pathlib import Path
from os import path
from typing import Any
from unittest.mock import patch

import pandas as pd
import responses


from opentutor_classifier import (  # type: ignore
    ArchLesson,
    DataDao,
    QuestionConfig,
    QuestionConfigSaveReq,
    ModelRef,
    ModelSaveReq,
    TrainingInput,
)
from opentutor_classifier.api import get_graphql_endpoint  # type: ignore
from opentutor_classifier.dao import FileDataDao  # type: ignore

# TODO: both opentutor-classifier and opentutor-classifier-api
# use this mocking code for tests, but we should not include
# it in the (non-test) dist of opentutor-classifier.
# We should instead make it its own installable pip module


def mock_gql_response(lesson: str, data_root: str, is_default_model=False):
    cfile = Path(path.join(data_root, lesson, "config.yaml"))
    dfile = Path(path.join(data_root, lesson, "training.csv"))
    training_data_prop = "allTrainingData" if is_default_model else "trainingData"
    res = {
        "data": {
            "me": {
                training_data_prop: {
                    "config": cfile.read_text() if cfile.is_file() else None,
                    "training": dfile.read_text() if dfile.is_file() else None,
                }
            }
        }
    }
    responses.add(responses.POST, get_graphql_endpoint(), json=res, status=200)


class _TestDataDao(DataDao):
    """
    Wrapper DataDao for tests.
    We need this because if the underlying DataDao is Gql,
    then after DataDao::save_config is called,
    we need to add a new mocked graphql response with the updated features
    """

    def __init__(self, dao: FileDataDao, is_default_model=False):
        self.dao = dao
        self.is_default_model = is_default_model

    @property
    def data_root(self) -> str:
        return self.dao.data_root

    @property
    def model_root(self) -> str:
        return self.dao.model_root

    def get_model_root(self, lesson: ArchLesson) -> str:
        return self.dao.get_model_root(lesson)

    def find_default_training_data(self) -> pd.DataFrame:
        return self.dao.find_default_training_data()

    def find_prediction_config(self, lesson: ArchLesson) -> QuestionConfig:
        return self.dao.find_prediction_config(lesson)

    def find_training_config(self, lesson: str) -> QuestionConfig:
        return self.dao.find_training_config(lesson)

    def find_training_input(self, lesson: str) -> TrainingInput:
        return self.dao.find_training_input(lesson)

    def load_pickle(self, ref: ModelRef) -> Any:
        return self.dao.load_pickle(ref)

    def trained_model_exists(self, ref: ModelRef) -> bool:
        return self.dao.trained_model_exists(ref)

    def save_config(self, req: QuestionConfigSaveReq) -> None:
        self.dao.save_config(req)
        mock_gql_response(
            req.lesson,
            self.dao.data_root,
            is_default_model=self.is_default_model,
        )

    def save_pickle(self, req: ModelSaveReq) -> None:
        self.dao.save_pickle(req)


@contextmanager
def mocked_data_dao(
    lesson: str,
    data_root: str,
    model_root: str,
    deployed_model_root: str,
    is_default_model=False,
):
    patcher = patch("opentutor_classifier.dao.find_data_dao")
    try:
        mock_gql_response(
            lesson,
            data_root,
            is_default_model=is_default_model,
        )
        mock_find_data_dao = patcher.start()
        mock_find_data_dao.return_value = _TestDataDao(
            FileDataDao(
                data_root,
                model_root=model_root,
                deployed_model_root=deployed_model_root,
            ),
            is_default_model,
        )
        yield None
    finally:
        patcher.stop()
