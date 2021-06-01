from itertools import combinations
from typing import Dict, List, Tuple

from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances


from .features import (
    number_of_negatives,
    word2vec_example_similarity,
)


class CustomAgglomerativeClustering:
    def __init__(self, word2vec: Word2VecKeyedVectors, index2word_set):
        self.word2vec = word2vec
        self.index2word_set = index2word_set
        self.word_alignment_dp: Dict[
            Tuple[Tuple[str, ...], Tuple[str, ...]], float
        ] = dict()

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

    def alignment_metric(self, x, y) -> float:
        if len(self.data[int(x[0])]) > len(self.data[int(y[0])]):
            x, y = y, x
        return 1 - self.word_alignment_feature(
            self.data[int(y[0])], self.data[int(x[0])]
        )

    def fit_predict(self, data: np.ndarray):
        self.data = data
        x = np.arange(len(self.data)).reshape(-1, 1)

        # Calculate pairwise distances with the new metric.
        m = pairwise_distances(x, x, metric=self.alignment_metric)

        agg = AgglomerativeClustering(
            n_clusters=2, affinity="precomputed", linkage="average"
        )
        return agg.fit_predict(m)

    def get_clusters(self, good_answers: np.array, bad_answers: np.array):
        good_labels = self.fit_predict(good_answers)
        bad_labels = self.fit_predict(bad_answers)
        return good_labels, bad_labels

    def get_best_candidate(
        self,
        sentence_cluster: np.array,
        word2vec: Word2VecKeyedVectors,
        index2word_set,
        cuttoff_length: int = 20,
    ) -> str:
        sentence_cluster = sentence_cluster[
            np.vectorize(lambda x: len(x) < cuttoff_length)(sentence_cluster)
        ]
        if len(sentence_cluster) < 5:
            return ""
        avg_proximity = np.zeros(len(sentence_cluster))
        for i, row1 in enumerate(sentence_cluster):
            for row2 in sentence_cluster:
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

        avg_proximity /= len(sentence_cluster)
        best_idx = np.argmax(avg_proximity)
        return sentence_cluster[best_idx]

    @staticmethod
    def generate_patterns_from_candidates(
        data: pd.DataFrame, best_targets: List[Tuple[str, str]]
    ):
        useful_pattern_for_each_cluster: Dict[str, List[str]] = {
            "good": list(),
            "bad": list(),
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
        self,
        good_answers: np.array,
        bad_answers: np.array,
        word2vec: Word2VecKeyedVectors,
        index2word_set,
        ideal_answer: Tuple[str],
    ):
        good_answers, bad_answers = np.array(good_answers), np.array(bad_answers)
        good_labels, bad_labels = self.get_clusters(good_answers, bad_answers)
        best_candidates = []
        best_candidates.append(
            (
                "good",
                self.get_best_candidate(
                    good_answers[good_labels == 0], word2vec, index2word_set
                ),
            )
        )
        best_candidates.append(
            (
                "good",
                self.get_best_candidate(
                    good_answers[good_labels == 1], word2vec, index2word_set
                ),
            )
        )
        best_candidates.append(
            (
                "bad",
                self.get_best_candidate(
                    bad_answers[bad_labels == 0], word2vec, index2word_set
                ),
            )
        )
        best_candidates.append(
            (
                "bad",
                self.get_best_candidate(
                    bad_answers[bad_labels == 1], word2vec, index2word_set
                ),
            )
        )

        data = pd.DataFrame(
            {
                "[SENTENCES]": list(good_answers) + list(bad_answers),
                "[LABELS]": [1] * len(good_answers) + [0] * len(bad_answers),
            }
        )
        data, candidates = self.generate_patterns_from_candidates(data, best_candidates)
        return data, candidates

    @staticmethod
    def deduplicate_patterns(
        patterns_with_fpr: List[Tuple[str, float]], fpr_cuttoff: float
    ) -> List[str]:
        fpr_store: Dict[str, float] = dict()
        features: List[str] = []
        for pattern, fpr in patterns_with_fpr:
            if fpr < fpr_cuttoff:
                continue
            ok = True
            for word in pattern.split("+"):
                word = word.strip()
                if fpr_store.get(word, float("-inf")) >= fpr:
                    ok = False
            fpr_store[pattern] = fpr
            if ok:
                features.append(pattern)
        features.sort()
        return features

    @staticmethod
    def select_feature_candidates(
        data: pd.DataFrame,
        candidates: Dict[str, List[str]],
        fpr_cuttoff: float = 0.98,
    ) -> Dict[str, List[str]]:
        useful_features: Dict[str, List[str]] = dict()
        for label in ("good", "bad"):
            good, bad, patterns = [], [], []
            for candidate in candidates[label]:
                good.append(np.sum(data[candidate] * data["[LABELS]"]))
                bad.append(np.sum(data[candidate] * (1 - data["[LABELS]"])))
                patterns.append(str(candidate))
            good, bad = np.array(good), np.array(bad)
            one_fpr = None
            if label == "good":
                one_fpr = 1 - (bad / np.sum(1 - data["[LABELS]"]))
            else:
                one_fpr = 1 - (good / np.sum(data["[LABELS]"]))

            patterns_with_fpr = list(zip(patterns, one_fpr))
            patterns_with_fpr.sort(key=lambda x: len(x[0]))
            # ignores bigger pattern if indivudal words in pattern have higher (1-fpr)
            useful_features[label] = CustomAgglomerativeClustering.deduplicate_patterns(
                patterns_with_fpr, fpr_cuttoff
            )

        return useful_features
