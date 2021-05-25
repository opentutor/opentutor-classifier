#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import environ

import pylru
from opentutor_classifier import (
    ClassifierConfig,
    ClassifierFactory,
    AnswerClassifier,
    ARCH_DEFAULT,
)


class Entry:
    def __init__(self, classifier: AnswerClassifier):
        self.classifier = classifier
        self.last_trained_at = self.classifier.get_last_trained_at()


class ClassifierDao:
    def __init__(self):
        self.cache = pylru.lrucache(int(environ.get("CACHE_MAX_SIZE", "100")))

    def find_classifier(
        self, config: ClassifierConfig, arch: str = ARCH_DEFAULT
    ) -> AnswerClassifier:
        if config.model_name in self.cache:
            e = self.cache[config.model_name]
            if e and e.last_trained_at >= e.classifier.get_last_trained_at():
                return e.classifier
        c = ClassifierFactory().new_classifier(config=config, arch=arch)

        self.cache[config.model_name] = Entry(c)
        return c
