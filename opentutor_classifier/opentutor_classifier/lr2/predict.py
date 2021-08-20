#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from opentutor_classifier.utils import model_last_updated_at, prop_bool
from os import path
from typing import Dict, List, Optional, Tuple

from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn import linear_model

from opentutor_classifier import (
    ARCH_LR2_CLASSIFIER,
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
from .constants import (
    ARCHETYPE_BAD,
    ARCHETYPE_GOOD,
    BAD,
    GOOD,
    MODEL_FILE_NAME,
    FEATURE_ARCHETYPE_ENABLED,
    FEATURE_PATTERNS_ENABLED,
    PATTERNS_BAD,
    PATTERNS_GOOD,
)
from .clustering_features import CustomDBScanClustering
from .dtos import ExpectationToEvaluate, InstanceModels
from .expectations import LRExpectationClassifier
from .features import preprocess_sentence
from opentutor_classifier.word2vec import find_or_load_word2vec


def _confidence_score(
    model: linear_model.LogisticRegression, sentence: List[List[float]]
) -> float:
    return model.predict_proba(sentence)[0, 1]


ModelAndConfig = Tuple[Dict[str, linear_model.LogisticRegression], QuestionConfig]


class LRAnswerClassifier(AnswerClassifier):
    def __init__(self):
        self._word2vec = None
        self._instance_models: Optional[InstanceModels] = None
        self.speech_act_classifier = SpeechActClassifier()
        self._model_and_config: ModelAndConfig = None
        self._is_default = False

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
                    arch=ARCH_LR2_CLASSIFIER,
                    lesson=self.model_name,
                    filename=MODEL_FILE_NAME,
                ),
                self.dao,
            )
            self._model_and_config = (cm.model, cm.config)
            self._is_default = cm.is_default
        return self._model_and_config

    def find_model_for_expectation(
        self,
        m_by_e: Dict[str, linear_model.LogisticRegression],
        expectation: str,
        return_first_model_if_only_one=False,
    ) -> linear_model.LogisticRegression:
        if expectation in m_by_e:
            return m_by_e[expectation]
        elif return_first_model_if_only_one and len(m_by_e) == 1:
            key = list(m_by_e.keys())[0]
            return m_by_e[key]
        else:
            return m_by_e[expectation]

    def find_word2vec(self) -> Word2VecKeyedVectors:
        if not self._word2vec:
            self._word2vec = find_or_load_word2vec(
                path.join(self.shared_root, "word2vec.bin")
            )
        return self._word2vec

    def find_score_and_class(
        self, classifier, exp_num_i: str, sent_features: List[List[float]]
    ):
        _evaluation = "Good" if classifier.predict(sent_features)[0] == 1 else "Bad"
        _score = _confidence_score(classifier, sent_features)
        return ExpectationClassifierResult(
            expectation_id=exp_num_i,
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
                if answer.expectation != ""
                else conf.get_all_expectation_names()
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
        clustering = CustomDBScanClustering(word2vec, index2word)
        for exp in expectations:
            exp_conf = conf.get_expectation(exp.expectation)
            sent_features = LRExpectationClassifier.calculate_features(
                question_proc,
                answer.input_sentence,
                sent_proc,
                preprocess_sentence(exp_conf.ideal),
                word2vec,
                index2word,
                exp_conf.features.get(GOOD) or [],
                exp_conf.features.get(BAD) or [],
                clustering,
                mode=ClassifierMode.PREDICT,
                feature_archetype_enabled=prop_bool(
                    FEATURE_ARCHETYPE_ENABLED, exp_conf.features
                ),
                feature_patterns_enabled=prop_bool(
                    FEATURE_PATTERNS_ENABLED, exp_conf.features
                ),
                expectation_config=conf.get_expectation(exp.expectation),
                patterns=exp_conf.features.get(PATTERNS_GOOD, [])
                + exp_conf.features.get(PATTERNS_BAD, [])
                if not self._is_default
                else [],
                archetypes=exp_conf.features.get(ARCHETYPE_GOOD, [])
                + exp_conf.features.get(ARCHETYPE_BAD, [])
                if not self._is_default
                else [],
            )
            result.expectation_results.append(
                self.find_score_and_class(
                    exp.classifier, exp.expectation, [sent_features]
                )
            )
        return result

    def get_last_trained_at(self) -> float:
        return model_last_updated_at(
            ARCH_LR2_CLASSIFIER, self.model_name, self.model_roots, MODEL_FILE_NAME
        )
