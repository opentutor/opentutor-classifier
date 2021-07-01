#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from opentutor_classifier.utils import model_last_updated_at
from os import path
from typing import Dict, List, Optional, Tuple

from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np
from sklearn import linear_model

from opentutor_classifier import (
    ARCH_LR_CLASSIFIER,
    AnswerClassifier,
    AnswerClassifierInput,
    AnswerClassifierResult,
    ClassifierConfig,
    ExpectationClassifierResult,
    ModelRef,
    QuestionConfig,
    ClassifierMode,
)
from opentutor_classifier.dao import find_predicton_config_and_pickle
from opentutor_classifier.speechact import SpeechActClassifier
from .constants import MODEL_FILE_NAME
from .clustering_features import CustomAgglomerativeClustering
from .dtos import ExpectationToEvaluate, InstanceModels
from .expectations import LRExpectationClassifier
from .features import preprocess_sentence
from opentutor_classifier.word2vec import find_or_load_word2vec


def _confidence_score(
    model: linear_model.LogisticRegression, sentence: List[List[float]]
) -> float:
    return model.predict_proba(sentence)[0, 1]


ModelAndConfig = Tuple[Dict[int, linear_model.LogisticRegression], QuestionConfig]


class LRAnswerClassifier(AnswerClassifier):
    def __init__(self):
        self._word2vec = None
        self._instance_models: Optional[InstanceModels] = None
        self.speech_act_classifier = SpeechActClassifier()
        self._model_and_config: ModelAndConfig = None

    def configure(
        self,
        config: ClassifierConfig,
    ) -> AnswerClassifier:
        self.dao = config.dao
        self.model_name = config.model_name
        self.model_roots = config.model_roots
        self.shared_root = config.shared_root
        return self

    @property
    def model_and_config(self) -> ModelAndConfig:
        if not self._model_and_config:
            cm = find_predicton_config_and_pickle(
                ModelRef(
                    arch=ARCH_LR_CLASSIFIER,
                    lesson=self.model_name,
                    filename=MODEL_FILE_NAME,
                ),
                self.dao,
            )
            self._model_and_config = (cm.model, cm.config)
        return self._model_and_config

    def find_model_for_expectation(
        self,
        m_by_e: Dict[int, linear_model.LogisticRegression],
        expectation: int,
        return_first_model_if_only_one=False,
    ) -> linear_model.LogisticRegression:
        return (
            m_by_e[0]
            if expectation >= len(m_by_e) and return_first_model_if_only_one
            else m_by_e[expectation]
        )

    def find_word2vec(self) -> Word2VecKeyedVectors:
        if not self._word2vec:
            self._word2vec = find_or_load_word2vec(
                path.join(self.shared_root, "word2vec.bin")
            )
        return self._word2vec

    def find_score_and_class(
        self, classifier, exp_num_i: int, sent_features: np.ndarray
    ):
        _evaluation = "Good" if classifier.predict(sent_features)[0] == 1 else "Bad"
        _score = _confidence_score(classifier, sent_features)
        return ExpectationClassifierResult(
            expectation=exp_num_i,
            evaluation=_evaluation,
            score=_score if _evaluation == "Good" else 1 - _score,
        )

    def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        sent_proc = preprocess_sentence(answer.input_sentence)
        m_by_e, conf = self.model_and_config
        expectations = [
            ExpectationToEvaluate(
                expectation=i,
                classifier=self.find_model_for_expectation(
                    m_by_e, i, return_first_model_if_only_one=True
                ),
            )
            for i in (
                [answer.expectation]
                if answer.expectation != -1
                else range(len(conf.expectations))
            )
        ]
        result = AnswerClassifierResult(input=answer, expectation_results=[])
        word2vec = self.find_word2vec()
        index2word = set(word2vec.index_to_key)
        result.speech_acts[
            "metacognitive"
        ] = self.speech_act_classifier.check_meta_cognitive(result)
        result.speech_acts["profanity"] = self.speech_act_classifier.check_profanity(
            result
        )
        question_proc = preprocess_sentence(conf.question)
        clustering = CustomAgglomerativeClustering(word2vec, index2word)
        for exp in expectations:
            exp_conf = conf.expectations[exp.expectation]
            sent_features = LRExpectationClassifier.calculate_features(
                question_proc,
                answer.input_sentence,
                sent_proc,
                preprocess_sentence(exp_conf.ideal),
                word2vec,
                index2word,
                exp_conf.features.get("good") or [],
                exp_conf.features.get("bad") or [],
                clustering,
                mode=ClassifierMode.PREDICT,
                expectation_config=conf.expectations[exp.expectation],
                patterns=exp_conf.features.get("patterns_good", [])
                + exp_conf.features.get("patterns_bad", [])
                or [],
                archetypes=exp_conf.features.get("cluster_archetypes", []),
            )
            result.expectation_results.append(
                self.find_score_and_class(
                    exp.classifier, exp.expectation, [sent_features]
                )
            )
        return result

    def get_last_trained_at(self) -> float:
        return model_last_updated_at(
            ARCH_LR_CLASSIFIER, self.model_name, self.model_roots, MODEL_FILE_NAME
        )
