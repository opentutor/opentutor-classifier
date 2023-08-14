#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

import pandas as pd
from opentutor_classifier import (
    AnswerClassifierTraining,
    TrainingConfig,
    DataDao,
    TrainingInput,
    TrainingResult,
)
from opentutor_classifier.lr2.train import LRAnswerClassifierTraining


class CompositeAnswerClassifierTraining(AnswerClassifierTraining):

    lr_training: AnswerClassifierTraining = LRAnswerClassifierTraining()

    def configure(self, config: TrainingConfig) -> "AnswerClassifierTraining":
        self.lr_training = self.lr_training.configure(config)
        return self

    def train(self, train_input: TrainingInput, dao: DataDao) -> TrainingResult:
        return self.lr_training.train(train_input, dao)

    def train_default(self, data: pd.DataFrame, dao: DataDao) -> TrainingResult:
        return self.lr_training.train_default(data, dao)
