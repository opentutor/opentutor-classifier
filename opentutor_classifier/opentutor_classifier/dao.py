#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from dataclasses import dataclass
import pickle
from os import environ, makedirs, path
import shutil
import tempfile
from typing import Any

import pandas as pd

from . import (
    ArchFile,
    ArchLesson,
    DataDao,
    ModelRef,
    ModelSaveReq,
    QuestionConfig,
    QuestionConfigSaveReq,
    TrainingInput,
)
from .api import (
    fetch_all_training_data,
    fetch_config,
    fetch_training_data,
    update_features,
    update_last_trained_at,
)
from .utils import load_data, load_config


MODEL_ROOT_DEFAULT = "./models"
MODELS_DEPLOYED_ROOT_DEFAULT = "./models_deployed"


def _get_model_root() -> str:
    return environ.get("MODEL_ROOT") or MODEL_ROOT_DEFAULT


def _get_deployed_model_root() -> str:
    return environ.get("MODEL_DEPLOYED_ROOT") or MODELS_DEPLOYED_ROOT_DEFAULT


_CONFIG_YAML = "config.yaml"
_TRAINING_CSV = "training.csv"


class FileDataDao(DataDao):
    def __init__(
        self, data_root: str, model_root: str = "", deployed_model_root: str = ""
    ):
        self._data_root = data_root
        self._model_root = model_root or self.data_root
        self._deployed_model_root = deployed_model_root

    @property
    def data_root(self) -> str:
        return self._data_root

    @property
    def model_root(self) -> str:
        return self._model_root

    @property
    def deployed_model_root(self) -> str:
        return self._deployed_model_root

    def get_model_root(self, lesson: ArchLesson) -> str:
        return path.join(self.model_root, lesson.arch, lesson.lesson)

    def _get_data_file(self, lesson, fname: str) -> str:
        return path.join(self.data_root, lesson, fname)

    def _get_model_file(self, ref: ModelRef) -> str:
        return path.join(
            self.get_model_root(ArchLesson(arch=ref.arch, lesson=ref.lesson)),
            ref.filename,
        )

    def _find_model_file(self, ref: ModelRef, raise_error_on_missing=True) -> str:
        m = self._get_model_file(ref)
        if path.isfile(m):
            return m
        if not self.deployed_model_root:
            if raise_error_on_missing:
                raise Exception(f"No model file found at {m}")
            else:
                return ""
        md = path.join(self.deployed_model_root, ref.arch, ref.lesson, ref.filename)
        if path.isfile(md):
            return md
        if raise_error_on_missing:
            raise Exception(f"No model file found at {m} or {md}")
        else:
            return ""

    def _setup_tmp(self, fname: str) -> str:
        tmp_save_dir = tempfile.mkdtemp()
        return path.join(tmp_save_dir, fname)

    def _replace(self, f_old: str, f_new) -> None:
        makedirs(path.dirname(path.abspath(f_new)), exist_ok=True)
        shutil.copyfile(f_old, f_new)

    def find_default_training_data(self) -> pd.DataFrame:
        return load_data(
            self._get_data_file(self.get_default_lesson_name(), _TRAINING_CSV)
        )

    def find_prediction_config(self, lesson: ArchLesson) -> QuestionConfig:
        return load_config(
            self._find_model_file(
                ModelRef(arch=lesson.arch, lesson=lesson.lesson, filename=_CONFIG_YAML)
            )
        )

    def find_training_config(self, lesson: str) -> QuestionConfig:
        return load_config(self._get_data_file(lesson, _CONFIG_YAML))

    def find_training_input(self, lesson: str) -> TrainingInput:
        return TrainingInput(
            lesson=lesson,
            config=self.find_training_config(lesson),
            data=load_data(self._get_data_file(lesson, _TRAINING_CSV)),
        )

    def load_pickle(self, ref: ModelRef) -> Any:
        with open(self._find_model_file(ref), "rb") as f:
            return pickle.load(f)

    def trained_model_exists(self, ref: ModelRef) -> bool:
        return path.isfile(self._get_model_file(ref))

    def remove_trained_model(self, ref: ArchLesson) -> None:
        shutil.rmtree(self.get_model_root(ref))

    def save_config(self, req: QuestionConfigSaveReq) -> None:
        tmpf = self._setup_tmp(_CONFIG_YAML)
        req.config.write_to(tmpf)
        self._replace(
            tmpf,
            self._get_model_file(
                ModelRef(arch=req.arch, lesson=req.lesson, filename=_CONFIG_YAML)
            ),
        )

    def save_pickle(self, req: ModelSaveReq) -> None:
        tmpf = self._setup_tmp(req.filename)
        with open(tmpf, "wb") as f:
            pickle.dump(req.model, f)
        self._replace(tmpf, self._get_model_file(req))


class WebAppDataDao(DataDao):
    """
    saves/loads config from both file system and gql as required by the web app.

    - find_prediction_config: reads QuestionConfig from (model_root) file,
                    because generated features MUST be exactly as saved

    - find_training_input: reads QuestionConfig and training data from
                        GQL, because we want to use whatever people have created w admin interface

    - save_config: saves the full question config to model root as a yaml file,
                and also saves any expectation features back to gql
    """

    def __init__(self, model_root=""):
        self.file_dao = FileDataDao(
            model_root or _get_model_root(),
            deployed_model_root=_get_deployed_model_root(),
        )

    @property
    def data_root(self) -> str:
        return self.file_dao.data_root

    @property
    def model_root(self) -> str:
        return self.file_dao.model_root

    def get_model_root(self, lesson: ArchLesson) -> str:
        return self.file_dao.get_model_root(lesson)

    def find_default_training_data(self) -> pd.DataFrame:
        return fetch_all_training_data()

    def find_prediction_config(self, lesson: ArchLesson) -> QuestionConfig:
        return self.file_dao.find_prediction_config(lesson)

    def find_training_config(self, lesson: str) -> QuestionConfig:
        return fetch_config(lesson)

    def find_training_input(self, lesson: str) -> TrainingInput:
        return fetch_training_data(lesson)

    def load_pickle(self, ref: ModelRef) -> Any:
        return self.file_dao.load_pickle(ref)

    def trained_model_exists(self, ref: ModelRef) -> bool:
        return self.file_dao.trained_model_exists(ref)

    def remove_trained_model(self, ref: ArchLesson) -> None:
        return self.file_dao.remove_trained_model(ref)

    def save_config(self, req: QuestionConfigSaveReq) -> None:
        if not req.skip_feature_update:
            update_features(req)
        update_last_trained_at(req.lesson)
        self.file_dao.save_config(req)

    def save_pickle(self, req: ModelSaveReq) -> None:
        self.file_dao.save_pickle(req)


def find_data_dao() -> DataDao:
    return WebAppDataDao()


@dataclass
class ConfigAndModel:
    config: QuestionConfig
    model: Any
    is_default: bool


def find_predicton_config_and_pickle(ref: ModelRef, dao: DataDao) -> ConfigAndModel:
    """
    Utility finder for the common prediction-time case
    of needing to load a single, pickle-file trained model
    and its sibling config.
    If the requested model has not been trained, returns default model
    """
    if dao.trained_model_exists(ref):
        return ConfigAndModel(
            config=dao.find_prediction_config(ref),
            model=dao.load_pickle(ref),
            is_default=False,
        )
    else:
        return ConfigAndModel(
            config=dao.find_training_config(ref.lesson),
            model=dao.load_default_pickle(
                ArchFile(arch=ref.arch, filename=ref.filename)
            ),
            is_default=True,
        )
