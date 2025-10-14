#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from opentutor_classifier.constants import DEPLOYMENT_MODE_OFFLINE
from opentutor_classifier.utils import model_last_updated_at, prop_bool
from os import path, environ
from typing import Dict, List, Optional, Tuple, Any

from sklearn import linear_model

from opentutor_classifier import (
    DEFAULT_LESSON_NAME,
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
from opentutor_classifier.word2vec_wrapper import Word2VecWrapper, get_word2vec
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
from opentutor_classifier.config import EVALUATION_BAD, EVALUATION_GOOD
from opentutor_classifier.log import LOGGER

DEPLOYMENT_MODE = environ.get("DEPLOYMENT_MODE") or DEPLOYMENT_MODE_OFFLINE

log = LOGGER


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
        self._default_model_and_config: ModelAndConfig = None
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
    def default_model_and_config(self) -> ModelAndConfig:
        if not self._default_model_and_config:
            cm = find_predicton_config_and_pickle(
                ModelRef(
                    arch=ARCH_LR2_CLASSIFIER,
                    lesson=DEFAULT_LESSON_NAME,
                    filename=MODEL_FILE_NAME,
                ),
                self.dao,
            )
            self._default_model_and_config = (cm.model, cm.config)
        return self._default_model_and_config

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

    def batch_preload_save_config_and_model_features(
        self, conf: QuestionConfig, expectations, w2v: Word2VecWrapper
    ):
        """
        ONLINE USE ONLY
        preprocesses data and fetches their vectors from w2v in one batch to store in memory
        """
        if DEPLOYMENT_MODE == DEPLOYMENT_MODE_OFFLINE:
            return
        words_to_preload = [*preprocess_sentence(conf.question)]
        for exp in expectations:
            exp_conf = conf.get_expectation(exp.expectation)
            words_to_preload.extend(preprocess_sentence(exp_conf.ideal))
            words_to_preload.extend(exp_conf.features.get(ARCHETYPE_GOOD, []))
            words_to_preload.extend(exp_conf.features.get(ARCHETYPE_BAD, []))
            words_to_preload.extend(exp_conf.features.get(PATTERNS_GOOD, []))
            words_to_preload.extend(exp_conf.features.get(PATTERNS_BAD, []))
        w2v.get_feature_vectors(set(words_to_preload), True)

    def save_config_and_model(self, embedding: bool = True) -> Dict[str, Any]:
        m_by_e, conf = self.model_and_config
        default_m_by_e, default_conf = self.default_model_and_config
        expectations = [
            ExpectationToEvaluate(
                expectation=i,
                classifier=self.find_model_for_expectation(m_by_e, default_m_by_e, i),
            )
            for i in conf.get_all_expectation_names()
        ]

        if embedding:
            w2v = self.find_word2vec_slim()
            self.batch_preload_save_config_and_model_features(conf, expectations, w2v)
        question_proc = preprocess_sentence(conf.question)

        slim_embeddings: Dict[str, List[float]] = dict()
        config_dict: Dict[str, Any] = dict()

        for exp in expectations:
            exp_conf = conf.get_expectation(exp.expectation)
            if embedding:
                slim_embeddings.update(
                    self.update_slim_embeddings(
                        preprocess_sentence(exp_conf.ideal),
                        question_proc,
                        exp_conf.features.get(ARCHETYPE_GOOD, []),
                        exp_conf.features.get(ARCHETYPE_BAD, []),
                        exp_conf.features.get(PATTERNS_GOOD, []),
                        exp_conf.features.get(PATTERNS_BAD, []),
                    )
                )
            config_dict[exp.expectation] = dict()
            config_dict[exp.expectation]["weights_bias"] = [
                list(exp.classifier.coef_[0]),
                exp.classifier.intercept_[0],
            ]
            config_dict[exp.expectation]["ideal"] = exp_conf.ideal
            config_dict[exp.expectation]["archetype_bad"] = exp_conf.features.get(
                ARCHETYPE_BAD, []
            )
            config_dict[exp.expectation]["archetype_good"] = exp_conf.features.get(
                ARCHETYPE_GOOD, []
            )
            config_dict[exp.expectation]["regex_good"] = (
                exp_conf.features.get(GOOD) or []
            )
            config_dict[exp.expectation]["regex_bad"] = exp_conf.features.get(BAD) or []
            config_dict[exp.expectation]["featureRegexAggregateDisabled"] = (
                exp_conf.features.get("featureRegexAggregateDisabled")
            )
            config_dict[exp.expectation]["featureDbScanClustersArchetypeEnabled"] = (
                exp_conf.features.get("featureDbScanClustersArchetypeEnabled")
            )
            config_dict[exp.expectation]["featureDbScanClustersPatternsEnabled"] = (
                exp_conf.features.get("featureDbScanClustersPatternsEnabled")
            )
            config_dict[exp.expectation]["featureLengthRatio"] = exp_conf.features.get(
                "featureLengthRatio"
            )
            config_dict[exp.expectation]["patterns_bad"] = (
                exp_conf.features.get("patterns_bad") or []
            )
            config_dict[exp.expectation]["patterns_good"] = (
                exp_conf.features.get("patterns_good") or []
            )

        if embedding:
            config_dict["embedding"] = slim_embeddings
        return config_dict

    def find_model_for_expectation(
        self,
        m_by_e: Dict[str, linear_model.LogisticRegression],
        default_m_by_e: Dict[str, linear_model.LogisticRegression],
        expectation: str,
    ) -> linear_model.LogisticRegression:
        if expectation in m_by_e:
            return m_by_e[expectation]
        else:
            key = list(default_m_by_e.keys())[0]
            return default_m_by_e[key]

    def find_word2vec(self) -> Word2VecWrapper:
        if not self._word2vec:
            self._word2vec = get_word2vec(
                path.join(self.shared_root, "word2vec.bin"),
                path.join(self.shared_root, "word2vec_slim.bin"),
            )
        return self._word2vec

    def find_word2vec_slim(self) -> Word2VecWrapper:
        return self.find_word2vec()

    def find_score_and_class(
        self, classifier, exp_num_i: str, sent_features: List[List[float]]
    ):
        _evaluation = (
            EVALUATION_GOOD
            if classifier.predict(sent_features)[0] == 1
            else EVALUATION_BAD
        )
        _score = _confidence_score(classifier, sent_features)
        return ExpectationClassifierResult(
            expectation_id=exp_num_i,
            evaluation=_evaluation,
            score=_score if _evaluation == EVALUATION_GOOD else 1 - _score,
        )

    def batch_preload_evaluate_features(
        self,
        answer: AnswerClassifierInput,
        index2word,
        config: QuestionConfig,
        expectations,
    ):
        """
        ONLINE USE ONLY
        preprocesses data and fetches their vectors from w2v in one batch to store in memory
        """
        if DEPLOYMENT_MODE == DEPLOYMENT_MODE_OFFLINE:
            return
        final_list = []
        final_list.extend(preprocess_sentence(answer.input_sentence))
        final_list.extend(preprocess_sentence(config.question))
        word2vec = self.find_word2vec()
        for exp in expectations:
            exp_conf = config.get_expectation(exp.expectation)
            final_list.extend(preprocess_sentence(exp_conf.ideal))
        final_set = set(final_list).intersection(index2word)
        word2vec.get_feature_vectors(final_set)

    async def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        log.debug("entering lr2 classifier")
        sent_proc = preprocess_sentence(answer.input_sentence)
        m_by_e, conf = self.model_and_config
        default_m_by_e, default_conf = self.default_model_and_config
        print(conf.get_all_expectation_names())
        expectations = [
            ExpectationToEvaluate(
                expectation=i,
                classifier=self.find_model_for_expectation(m_by_e, default_m_by_e, i),
            )
            for i in (
                [answer.expectation]
                if answer.expectation != ""
                else conf.get_all_expectation_names()
            )
        ]
        result = AnswerClassifierResult(input=answer, expectation_results=[])
        word2vec = self.find_word2vec()
        index2word = set(word2vec.index_to_key(False))
        result.speech_acts["metacognitive"] = (
            self.speech_act_classifier.check_meta_cognitive(result)
        )
        result.speech_acts["profanity"] = self.speech_act_classifier.check_profanity(
            result
        )
        question_proc = preprocess_sentence(conf.question)
        clustering = CustomDBScanClustering(word2vec, index2word)

        self.batch_preload_evaluate_features(answer, index2word, conf, expectations)

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
                patterns=(
                    exp_conf.features.get(PATTERNS_GOOD, [])
                    + exp_conf.features.get(PATTERNS_BAD, [])
                    if not self._is_default
                    else []
                ),
                archetypes=(
                    exp_conf.features.get(ARCHETYPE_GOOD, [])
                    + exp_conf.features.get(ARCHETYPE_BAD, [])
                    if not self._is_default
                    else []
                ),
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

    def update_slim_embeddings(
        self,
        ideal: List[str],
        question: List[str],
        archtypes_good: List[str],
        archetypes_bad: List[str],
        patterns_good: List[str],
        patterns_bad: List[str],
    ):
        embeddings: Dict[str, List[float]] = dict()
        words_set = set()
        for word in ideal + question:
            words_set.add(word)
        for archetype in archtypes_good + archetypes_bad:
            for word in archetype.lower().split():
                words_set.add(word)
        for pattern in patterns_bad + patterns_good:
            for word in pattern.split(" + "):
                words_set.add(word)

        word_vecs = self._word2vec.get_feature_vectors(words_set, True)

        for word in word_vecs.keys():
            embeddings[word] = list(map(lambda x: round(float(x), 9), word_vecs[word]))
        return embeddings
