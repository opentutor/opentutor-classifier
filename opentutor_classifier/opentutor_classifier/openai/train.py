#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

import pandas as pd
import os
from typing import Any
from opentutor_classifier import (
    AnswerClassifierTraining,
    TrainingConfig,
    DataDao,
    TrainingInput,
    TrainingResult,
    ExpectationTrainingResult,
    QuestionConfigSaveReq,
)
from opentutor_classifier.openai import ARCH_OPENAI_CLASSIFIER
from opentutor_classifier.dao import ModelSaveReq, ArchLesson, MODEL_ROOT_DEFAULT
from opentutor_classifier.config import (
    LABEL_BAD,
    LABEL_GOOD,
    EVALUATION_BAD,
    EVALUATION_GOOD,
)
from opentutor_classifier.openai.shared import OpenAIGroundTruth, OpenAIGroundTruthEntry
from opentutor_classifier.openai.constants import GROUNDTRUTH_FILENAME
from typing import cast

MAX_NUMBER_OF_ANSWERS = 100


class OpenAIAnswerClassifierTraining(AnswerClassifierTraining):
    def configure(self, config: TrainingConfig) -> "AnswerClassifierTraining":
        return self

    def train(
        self, train_input: TrainingInput, dao: DataDao, developer_mode: bool = False
    ) -> TrainingResult:
        expectation_ids = [
            expectation.expectation_id
            for expectation in train_input.config.expectations
        ]
        unique_text = train_input.data.groupby("text", group_keys=True)

        training_json = OpenAIGroundTruth({})

        for group in unique_text.groups:
            group_frame = cast(pd.DataFrame, unique_text.get_group(group))

            if (
                len(group_frame.index) < len(expectation_ids)
                or len(training_json.training_answers) >= MAX_NUMBER_OF_ANSWERS
            ):
                continue

            entry = OpenAIGroundTruthEntry(answer_text=cast(str, group), concepts={})

            for index, row in group_frame.iterrows():
                if row["exp_num"] in expectation_ids and row["label"] in [
                    LABEL_GOOD,
                    LABEL_BAD,
                    EVALUATION_GOOD,
                    EVALUATION_BAD,
                ]:
                    entry.concepts[row["exp_num"]] = row["label"]

            if len(entry.concepts) == len(expectation_ids):
                training_json.training_answers[cast(str, group)] = entry

        dao.save_ground_truth(
            ModelSaveReq(
                arch=ARCH_OPENAI_CLASSIFIER,
                lesson=train_input.lesson,
                filename=GROUNDTRUTH_FILENAME,
                model=training_json.to_dict(),
            )
        )

        dao.save_config(
            QuestionConfigSaveReq(
                arch=ARCH_OPENAI_CLASSIFIER,
                lesson=train_input.lesson,
                config=train_input.config,
            )
        )

        expectation_results = [
            ExpectationTrainingResult(expectation_id=exp_id, accuracy=1.0)
            for exp_id in expectation_ids
        ]
        return TrainingResult(
            lesson=train_input.lesson,
            expectations=expectation_results,
            models=dao.get_model_root(
                ArchLesson(arch=ARCH_OPENAI_CLASSIFIER, lesson=train_input.lesson)
            ),
        )

    def train_default(self, data: pd.DataFrame, dao: DataDao) -> TrainingResult:
        raise NotImplementedError()

    def upload_model(self, s3: Any, lesson: str, s3_bucket: str):
        s3.upload_file(
            os.path.join(
                MODEL_ROOT_DEFAULT,
                ARCH_OPENAI_CLASSIFIER,
                lesson,
                GROUNDTRUTH_FILENAME,
            ),
            s3_bucket,
            os.path.join(lesson, ARCH_OPENAI_CLASSIFIER, GROUNDTRUTH_FILENAME),
        )
