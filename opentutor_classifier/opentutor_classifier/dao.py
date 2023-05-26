#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from dataclasses import dataclass
import pickle
import json
from os import environ, makedirs, path
import shutil
import tempfile
from typing import Any
from datetime import datetime

import pandas as pd

from .constants import DEPLOYMENT_MODE_OFFLINE

from . import (
    DEFAULT_LESSON_NAME,
    ArchFile,
    ArchLesson,
    DataDao,
    EmbeddingSaveReq,
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
from .utils import load_data, load_config, require_env
from .logger import get_logger

import boto3
import botocore

logger = get_logger("dao")
s3 = boto3.client("s3")
SHARED = environ.get("SHARED_ROOT") or "shared"
logger.info(f"shared: {SHARED}")
MODELS_BUCKET = require_env("MODELS_BUCKET")
logger.info(f"bucket: {MODELS_BUCKET}")

DEPLOYMENT_MODE = environ.get("DEPLOYMENT_MODE") or DEPLOYMENT_MODE_OFFLINE
logger.info(f"Deployment Mode: {DEPLOYMENT_MODE}")

# online mode lambdas only allow writing to /tmp folder
MODEL_ROOT_DEFAULT = (
    "/models" if DEPLOYMENT_MODE == DEPLOYMENT_MODE_OFFLINE else "/tmp/models"
)
MODELS_DEPLOYED_ROOT_DEFAULT = (
    "/models_deployed"
    if DEPLOYMENT_MODE == DEPLOYMENT_MODE_OFFLINE
    else "/tmp/models_deployed"
)
logger.info(f"model root default: {MODEL_ROOT_DEFAULT}")


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

    def _get_embedding_file(self, ref: ModelRef) -> str:
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

    def save_embeddings(self, req: EmbeddingSaveReq) -> None:
        tmpf = self._setup_tmp(req.filename)
        with open(tmpf, "w") as f:
            json.dump(req.embedding, f)
        self._replace(tmpf, self._get_embedding_file(req))


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
        if req.lesson != "default":
            update_last_trained_at(req.lesson)
        self.file_dao.save_config(req)

    def save_pickle(self, req: ModelSaveReq) -> None:
        self.file_dao.save_pickle(req)

    def save_embeddings(self, req: EmbeddingSaveReq) -> None:
        self.file_dao.save_embeddings(req)


def find_data_dao() -> DataDao:
    return WebAppDataDao()


@dataclass
class ConfigAndModel:
    config: QuestionConfig
    model: Any
    is_default: bool


def find_predicton_config_and_pickle(ref: ModelRef, dao: DataDao) -> ConfigAndModel:
    if DEPLOYMENT_MODE == DEPLOYMENT_MODE_OFFLINE:
        return find_predicton_config_and_pickle_offline(ref, dao)
    else:
        return find_predicton_config_and_pickle_online(ref, dao)


def find_predicton_config_and_pickle_offline(
    ref: ModelRef, dao: DataDao
) -> ConfigAndModel:
    """
    FOR OFFLINE MODE USE
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


def get_and_update_model_from_s3(ref: ModelRef, model_in_memory_exists: bool = True):
    """
    ONLINE USE ONLY
    model_in_memory_exists: only replace model in memory if model in s3 was updated
    """
    model_lambda_file_path = path.join(
        MODEL_ROOT_DEFAULT, ref.arch, ref.lesson, ref.filename
    )
    config_lambda_file_path = path.join(
        MODEL_ROOT_DEFAULT, ref.arch, ref.lesson, _CONFIG_YAML
    )
    if model_in_memory_exists:
        # Check if model exists in s3 that is more up to date than model in memory
        logger.info(f"model file exists in lambda memory: {model_lambda_file_path}")
        modified_time = path.getmtime(model_lambda_file_path)
        utc_mod_time = datetime.utcfromtimestamp(modified_time)
        logger.info(f"model file modified at {utc_mod_time}")
    try:
        model_s3_path = path.join(ref.lesson, ref.arch, ref.filename)
        logger.info(f"model s3 path: {model_s3_path}")
        model_from_s3 = s3.get_object(
            **{
                "Bucket": MODELS_BUCKET,
                "Key": model_s3_path,
                **(
                    {"IfModifiedSince": str(utc_mod_time)}
                    if model_in_memory_exists
                    else {}
                ),
            }
        )
        config_s3_path = path.join(ref.lesson, ref.arch, _CONFIG_YAML)
        logger.info(f"model s3 path: {config_s3_path}")

        config_from_s3 = s3.get_object(
            **{
                "Bucket": MODELS_BUCKET,
                "Key": config_s3_path,
                **(
                    {"IfModifiedSince": str(utc_mod_time)}
                    if model_in_memory_exists
                    else {}
                ),
            }
        )
        logger.info("model and config found in s3")
        # Update model and config in memory with up to date versions from s3
        makedirs(path.dirname(model_lambda_file_path), exist_ok=True)
        makedirs(path.dirname(config_lambda_file_path), exist_ok=True)
        with open(model_lambda_file_path, "wb") as f:
            for chunk in model_from_s3["Body"].iter_chunks(chunk_size=4096):
                f.write(chunk)
        with open(config_lambda_file_path, "wb") as f:
            for chunk in config_from_s3["Body"].iter_chunks(chunk_size=4096):
                f.write(chunk)
        logger.info("model file updated")
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey" and e.response["Error"]["Code"] != "304": # NoSuchKey indicates no trained model in s3, 304 indicates model exists but not updated
            logger.error(ref)
            logger.error(e)
            raise e
        logger.debug("model file not updated in s3 since last fetch")
        return False


def find_predicton_config_and_pickle_online(
    ref: ModelRef, dao: DataDao
) -> ConfigAndModel:
    """
    FOR ONLINE MODE USE
    Utility finder for the common prediction-time case
    of needing to load a single, pickle-file trained model
    and its sibling config.
    if the requested model is in memory, first checks if a more up to date model exists in s3
    if the requested model is not in memory, checks if a model exists in s3.
    else return default model
    """

    if dao.trained_model_exists(ref):
        logger.info(f"model in memory for {ref.lesson}, checking s3")
        get_and_update_model_from_s3(ref, model_in_memory_exists=True)
        return ConfigAndModel(
            config=dao.find_prediction_config(ref),
            model=dao.load_pickle(ref),
            is_default=False,
        )
    elif get_and_update_model_from_s3(ref, model_in_memory_exists=False):
        logger.info("model not in memory but got it from s3")
        return ConfigAndModel(
            config=dao.find_prediction_config(ref),
            model=dao.load_pickle(ref),
            is_default=False,
        )
    else:
        default_ref: ModelRef = ModelRef(
            filename=ref.filename, lesson=DEFAULT_LESSON_NAME, arch=ref.arch
        )
        logger.info(
            f"No model found for lesson {ref.lesson} in memory nor s3, using default model"
        )
        model_in_memory_exists = dao.trained_model_exists(default_ref)
        get_and_update_model_from_s3(default_ref, model_in_memory_exists)
        return ConfigAndModel(
            config=dao.find_training_config(ref.lesson),
            model=dao.load_default_pickle(
                ArchFile(arch=ref.arch, filename=ref.filename)
            ),
            is_default=True,
        )
