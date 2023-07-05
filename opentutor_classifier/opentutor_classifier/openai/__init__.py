#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from dataclasses import dataclass
from typing import List, Dict
from opentutor_classifier import ArchClassifierFactory, ClassifierConfig, AnswerClassifier, AnswerClassifierTraining, TrainingConfig, ARCH_OPENAI_CLASSIFIER, register_classifier_factory
from .train import OpenAIAnswerClassifierTraining
from .predict import OpenAIAnswerClassifier
import json

@dataclass
class OpenAICall:
    system_assignment: str
    user_concepts: List[str]
    user_answer: List[str]
    user_template: dict()
    user_guardrails: str

    def to_openai_json(self) -> str:
        result:dict = {}
        result["system-assignment"] = self.system_assignment
        user_concepts: dict = {}
        for index, concept in enumerate(self.user_concepts):
            user_concepts["Concept " + str(index)] = concept
        result["user-concepts"] = user_concepts
        user_answer: dict = {}
        for index, answer in enumerate(self.user_answer):
            user_answer["Answer " + str(index)] = {"Answer Text" : answer}        
        result["user-answer"] = user_answer
        result["user-template"] = self.user_template
        result["user-guardrails"] = self.user_guardrails
        return json.dumps(result)

@dataclass
class Concept:
    is_known: bool
    confidence: float
    justification: str

@dataclass
class Answer:
    answer_text: str
    concepts: Dict[str, Concept]

@dataclass
class OpenAIResultContent:
    answers: Dict[str, Answer]


class __ArchClassifierFactory(ArchClassifierFactory):
    def has_trained_model(self, lesson: str, config: ClassifierConfig) -> bool:
        return True

    def new_classifier(self, config: ClassifierConfig) -> AnswerClassifier:
        return OpenAIAnswerClassifier().configure(config)

    def new_classifier_default(self, config: ClassifierConfig) -> AnswerClassifier:
        raise Exception("OpenAI has no default model")

    def new_training(self, config: TrainingConfig) -> AnswerClassifierTraining:
        return OpenAIAnswerClassifierTraining().configure(config)


register_classifier_factory(ARCH_OPENAI_CLASSIFIER, __ArchClassifierFactory())

