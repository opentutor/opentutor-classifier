#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from typing import Any, Dict

from numpy import ndarray
from opentutor_classifier.opentutor_classifier.api import (
    sbert_word_to_vec,
    get_sbert_index_to_key,
)


class Word2VecWrapper:
    def get_feature_vectors(self, words, slim: bool = False) -> Dict[str, ndarray]:
        sbert_w2v_result = sbert_word_to_vec(words, slim)
        return sbert_w2v_result

    def index_to_key(self, slim: bool = False) -> Any:
        return get_sbert_index_to_key(slim)
