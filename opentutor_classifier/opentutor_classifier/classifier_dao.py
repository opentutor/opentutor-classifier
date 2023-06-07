#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from datetime import datetime, timezone
from opentutor_classifier.api import fetch_lesson_updated_at
from os import environ

import pylru
from opentutor_classifier import (
    get_classifier_arch,
    ClassifierConfig,
    ClassifierFactory,
    AnswerClassifier,
)


class Entry:
    def __init__(self, classifier: AnswerClassifier, lesson_updated_at: datetime):
        self.classifier = classifier
        self.last_trained_at = self.classifier.get_last_trained_at()
        self.lesson_updated_at = lesson_updated_at


class ClassifierDao:
    def __init__(self):
        self.cache = pylru.lrucache(int(environ.get("CACHE_MAX_SIZE", "100")))

    def find_classifier(
        self, lesson: str, config: ClassifierConfig, arch: str = ""
    ) -> AnswerClassifier:
        cfac = ClassifierFactory()
        lesson_updated_at = (
            fetch_lesson_updated_at(lesson)
            if cfac.has_trained_model(lesson, config, arch=arch)
            else datetime.min.replace(tzinfo=timezone.utc)
        )
        if config.model_name in self.cache:
            e = self.cache[config.model_name]
            if (
                e
                and e.last_trained_at >= e.classifier.get_last_trained_at()
                and e.lesson_updated_at >= lesson_updated_at
            ):
                return e.classifier
        c = cfac.new_classifier(config=config, arch=arch or get_classifier_arch())
        self.cache[config.model_name] = Entry(c, lesson_updated_at)
        return c
