#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from itertools import combinations
import heapq
from opentutor_classifier.lr2.constants import BAD, GOOD
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import SelectKBest, chi2

from text_to_num import alpha2digit

from opentutor_classifier.word2vec_wrapper import Word2VecWrapper

from .features import (
    number_of_negatives,
    word2vec_example_similarity,
    check_is_pattern_match,
    _avg_feature_vector,
)


class CustomDBScanClustering:
    def __init__(self, word2vec: Word2VecWrapper, index2word_set):
        self.word2vec = word2vec
        self.index2word_set = index2word_set
        self.word_alignment_dp: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float] = (
            dict()
        )

    def word_alignment_feature(self, example: List[str], ia: List[str]) -> float:
        key = (tuple(example), tuple(ia))
        if key in self.word_alignment_dp:
            return self.word_alignment_dp[key]

        cost = []
        n_exact_matches = len(set(ia).intersection(set(example)))
        ia_, example_ = (
            list(set(ia).difference(example)),
            list(set(example).difference(ia)),
        )
        if not ia_:
            return 1

        for ia_i in ia_:
            inner_cost = []
            for e in example_:
                dist = word2vec_example_similarity(
                    self.word2vec, self.index2word_set, [e], [ia_i]
                )
                inner_cost.append(dist)
            cost.append(inner_cost)
        row_idx, col_idx = linear_sum_assignment(cost, maximize=True)

        self.word_alignment_dp[key] = (
            n_exact_matches + sum([cost[r][c] for r, c in zip(row_idx, col_idx)])
        ) / float(len(ia_) + n_exact_matches)
        return self.word_alignment_dp[key]

    def fit_predict(self, data: List[np.ndarray]) -> np.ndarray:
        # Calculate pairwise distances with the new metric.'
        agg = DBSCAN(eps=0.30)
        return agg.fit_predict(data) if len(data) > 0 else []

    def get_embedding(self, sentence: List[str], num_features=300) -> np.ndarray:
        return _avg_feature_vector(
            words=sentence,
            model=self.word2vec,
            num_features=num_features,
            index2word_set=self.index2word_set,
        )

    def get_clusters(
        self, good_answers: np.ndarray, bad_answers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        good_labels = self.fit_predict(
            [self.get_embedding(example) for example in good_answers]
        )
        bad_labels = self.fit_predict(
            [self.get_embedding(example) for example in bad_answers]
        )
        return good_labels, bad_labels

    def get_best_candidate(
        self, sentence_cluster: np.ndarray, cuttoff_length: int = 20, batch_size=10
    ) -> List[str]:
        sentence_cluster = sentence_cluster[
            np.vectorize(lambda x: len(x) < cuttoff_length)(sentence_cluster)
        ]
        if len(sentence_cluster) < 5:
            return [""]

        final_candidates: List[List[str]] = list(sentence_cluster)
        total_sentences: int = len(final_candidates)

        while total_sentences != 1:
            for idx in range(0, total_sentences, batch_size):
                current_batch = final_candidates[idx : (idx + batch_size)]  # noqa E203
                avg_proximity = np.zeros(len(current_batch))
                for i, row1 in enumerate(current_batch):
                    for row2 in current_batch:
                        if row1 == row2:
                            continue
                        avg_proximity[i] += (
                            len(row2) / (len(row2) + len(row1))
                        ) * self.word_alignment_feature(
                            row2, row1
                        )  # check best alignment in both directions
                        avg_proximity[i] += (
                            len(row1) / (len(row2) + len(row1))
                        ) * self.word_alignment_feature(row1, row2)

                avg_proximity /= len(current_batch)
                best_idx = int(np.argmax(avg_proximity))
                final_candidates.append(current_batch[best_idx])
            final_candidates = final_candidates[total_sentences:]
            total_sentences = len(final_candidates)

        return final_candidates[0]

    @staticmethod
    def generate_patterns_from_candidates(
        data: pd.DataFrame, best_targets: List[Tuple[str, List[str]]]
    ):
        useful_pattern_for_each_cluster: Dict[str, List[str]] = {
            GOOD: list(),
            BAD: list(),
        }

        for label, best_target in best_targets:
            words = set()
            for word in best_target:
                if number_of_negatives([word])[0] > 0:
                    words.add("[NEG]")
                else:
                    words.add(word)

            for word in words:
                data[word] = data["[SENTENCES]"].apply(lambda x: 1 * (word in x))

            data["[NEG]"] = data["[SENTENCES]"].apply(
                lambda x: 1 * (number_of_negatives(x)[0] > 0)
            )

            # maximum pattern length will be 4
            total_patterns = list(words)
            for i in range(2, min(len(words) + 1, 3)):
                combs = combinations(words, i)
                for comb_ in combs:
                    comb = sorted(list(comb_))
                    data[" + ".join(comb)] = 1
                    for word in comb:
                        data[" + ".join(comb)] *= data[word]
                    total_patterns.append(" + ".join(comb))
            useful_pattern_for_each_cluster[label].extend(total_patterns)

        return data, useful_pattern_for_each_cluster

    def generate_feature_candidates(
        self, good_answers: np.ndarray, bad_answers: np.ndarray, train_quality: int
    ):
        good_answers, bad_answers = np.array(good_answers, dtype=object), np.array(
            bad_answers, dtype=object
        )
        good_labels, bad_labels = self.get_clusters(good_answers, bad_answers)

        best_candidates = []

        for cluster_label in np.unique(good_labels):
            best_candidates.append(
                (
                    GOOD,
                    self.get_best_candidate(
                        good_answers[good_labels == cluster_label],
                    ),
                )
            )

        for cluster_label in np.unique(bad_labels):
            best_candidates.append(
                (
                    BAD,
                    self.get_best_candidate(
                        bad_answers[bad_labels == cluster_label],
                    ),
                )
            )

        data = pd.DataFrame(
            {
                "[SENTENCES]": list(good_answers) + list(bad_answers),
                "[LABELS]": [1] * len(good_answers) + [0] * len(bad_answers),
            }
        )
        archetype = {
            GOOD: [
                " ".join(archetype)
                for label, archetype in best_candidates
                if label == GOOD and archetype != [""]
            ],
            BAD: [
                " ".join(archetype)
                for label, archetype in best_candidates
                if label == BAD and archetype != [""]
            ],
        }

        if train_quality > 1:
            data, candidates = self.generate_patterns_from_candidates(
                data, best_candidates
            )
            return data, candidates, archetype
        else:
            return archetype

    @staticmethod
    def deduplicate_patterns(
        patterns_with_fpr: List[Tuple[str, float]],
        fpr_cuttoff: float,
        top_n=20,
    ) -> List[str]:
        fpr_store: Dict[str, float] = dict()
        features: List[Tuple[float, str]] = []
        for pattern, fpr in sorted(list(set(patterns_with_fpr))):
            if fpr <= fpr_cuttoff:
                continue
            ok = True
            for word in pattern.split("+"):
                word = word.strip()
                if fpr_store.get(word, float("-inf")) >= fpr:
                    ok = False
            fpr_store[pattern] = fpr
            if ok:
                features.append((fpr, pattern))

        top_features = top_features = [
            pat for _, pat in heapq.nsmallest(top_n, features)
        ]
        top_features.sort()
        return top_features

    @staticmethod
    def univariate_feature_selection(
        patterns: List[str], input_x: List[str], input_y: List[str], n: int = 10
    ) -> List[str]:
        if len(patterns) <= n:
            return patterns
        train_x: List[List[int]] = []
        train_y: List[int] = [1 * (x == GOOD) for x in input_y]

        for raw_example in input_x:
            raw_example = alpha2digit(raw_example, "en")
            feat: List[int] = []
            for pattern in patterns:
                feat.append(check_is_pattern_match(raw_example, pattern))
            train_x.append(feat)

        skb = SelectKBest(chi2, k=min(n, len(patterns)))
        skb.fit(train_x, train_y)
        masks: List[bool] = skb.get_support()
        return [pattern for mask, pattern in zip(masks, patterns) if mask]

    @staticmethod
    def select_feature_candidates(
        data: pd.DataFrame,
        candidates: Dict[str, List[str]],
        input_x: List[str],
        input_y: List[str],
        fpr_cuttoff: float = 0.80,
    ) -> Dict[str, List[str]]:

        useful_features: Dict[str, List[str]] = dict()
        for label in (GOOD, BAD):
            good, bad, patterns = [], [], []
            for candidate in candidates[label]:
                good.append(np.sum(data[candidate] * data["[LABELS]"]))
                bad.append(np.sum(data[candidate] * (1 - data["[LABELS]"])))
                patterns.append(str(candidate))
            one_fpr = None
            if label == GOOD:
                one_fpr = 1 - (
                    np.array(good, dtype=object) / np.sum(1 - data["[LABELS]"])
                )
            else:
                one_fpr = 1 - (np.array(bad, dtype=object) / np.sum(data["[LABELS]"]))

            patterns_with_fpr = list(zip(patterns, one_fpr))
            patterns_with_fpr.sort(key=lambda x: len(x[0]))
            # ignores bigger pattern if indivudal words in pattern have higher (1-fpr)
            useful_features[label] = CustomDBScanClustering.deduplicate_patterns(
                patterns_with_fpr, fpr_cuttoff
            )
            useful_features[label] = (
                CustomDBScanClustering.univariate_feature_selection(
                    useful_features[label], input_x, input_y
                )
            )

        return useful_features
