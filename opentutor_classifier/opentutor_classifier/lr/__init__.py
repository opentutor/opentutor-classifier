#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path

from opentutor_classifier import (
    ARCH_LR_CLASSIFIER,
    AnswerClassifier,
    AnswerClassifierTraining,
    ArchClassifierFactory,
    ClassifierConfig,
    TrainingConfig,
    register_classifier_factory,
)
from .predict import LRAnswerClassifier, LRExpectationClassifier  # noqa: F401
from opentutor_classifier.word2vec import find_or_load_word2vec  # noqa: F401
from .train import LRAnswerClassifierTraining  # noqa: F401


class __ArchClassifierFactory(ArchClassifierFactory):
    def new_classifier(self, config: ClassifierConfig) -> AnswerClassifier:
        """
        TODO: LRAnswerClassifier needs to be updated
        to handle multiple model roots (copy from SVMAnswerClassifier)
        """
        return LRAnswerClassifier(
            # config.model_name,
            model_root=path.join(config.model_roots[0], config.model_name),
            shared_root=config.shared_root,
        )

    def new_classifier_default(
        self, config: ClassifierConfig, arch=""
    ) -> AnswerClassifier:
        raise NotImplementedError()

    def new_training(self, config: TrainingConfig) -> AnswerClassifierTraining:
        raise NotImplementedError()


register_classifier_factory(ARCH_LR_CLASSIFIER, __ArchClassifierFactory())
