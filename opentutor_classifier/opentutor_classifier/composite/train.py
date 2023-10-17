#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

import pandas as pd
from typing import Any
from opentutor_classifier import (
    AnswerClassifierTraining,
    TrainingConfig,
    DataDao,
    TrainingInput,
    TrainingResult,
)
from opentutor_classifier.lr2.train import LRAnswerClassifierTraining
from opentutor_classifier.openai.train import OpenAIAnswerClassifierTraining


class CompositeAnswerClassifierTraining(AnswerClassifierTraining):

    lr_training: AnswerClassifierTraining = LRAnswerClassifierTraining()
    openai_training: AnswerClassifierTraining = OpenAIAnswerClassifierTraining()

    def configure(self, config: TrainingConfig) -> "AnswerClassifierTraining":
        self.lr_training = self.lr_training.configure(config)
        self.openai_training = self.openai_training.configure(config)
        return self

    def train(
        self, train_input: TrainingInput, dao: DataDao, developer_mode: bool = False
    ) -> TrainingResult:
        self.openai_training.train(train_input, dao, developer_mode)
        return self.lr_training.train(train_input, dao, developer_mode)

    def train_default(self, data: pd.DataFrame, dao: DataDao) -> TrainingResult:
        return self.lr_training.train_default(data, dao)

    def upload_model(self, s3: Any, lesson: str, s3_bucket: str):
        self.openai_training.upload_model(s3, lesson, s3_bucket)
        self.lr_training.upload_model(s3, lesson, s3_bucket)
        return
