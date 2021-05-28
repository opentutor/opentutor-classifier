#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from opentutor_classifier import (
    ARCH_LR_EMB_DIFF_CLASSIFIER,
    AnswerClassifier,
    AnswerClassifierTraining,
    ArchClassifierFactory,
    ClassifierConfig,
    TrainingConfig,
    register_classifier_factory,
)
from .predict import LRAnswerClassifier
from .train import LRAnswerClassifierTraining


class __ArchClassifierFactory(ArchClassifierFactory):
    def new_classifier(self, config: ClassifierConfig) -> AnswerClassifier:
        return LRAnswerClassifier().configure(config)

    def new_classifier_default(self, config: ClassifierConfig) -> AnswerClassifier:
        if config.model_name != "default":
            raise Exception("model name for default classifier must be default")
        return LRAnswerClassifier().configure(config)

    def new_training(self, config: TrainingConfig) -> AnswerClassifierTraining:
        return LRAnswerClassifierTraining().configure(config)


register_classifier_factory(ARCH_LR_EMB_DIFF_CLASSIFIER, __ArchClassifierFactory())
