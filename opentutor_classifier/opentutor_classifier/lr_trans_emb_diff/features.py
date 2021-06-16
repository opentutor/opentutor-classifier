#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import re
from typing import List

from nltk import pos_tag
from nltk.tokenize import word_tokenize

from opentutor_classifier.stopwords import STOPWORDS
from text_to_num import alpha2digit

word_mapper = {
    "n't": "not",
}


def preprocess_punctuations(sentence: str) -> str:
    sentence = re.sub(r"[\-=]", " ", sentence)
    sentence = re.sub(r"[%]", " percent ", sentence)
    sentence = re.sub("n't", " not", sentence)
    sentence = re.sub(r"[()~!^,?.\'$]", "", sentence)
    return sentence


def preprocess_sentence(sentence: str) -> List[str]:
    sentence = preprocess_punctuations(sentence.lower())
    sentence = alpha2digit(sentence, "en")
    word_tokens_groups: List[str] = [
        word_tokenize(entry)
        for entry in ([sentence] if isinstance(sentence, str) else sentence)
    ]
    result_words = []
    for entry in word_tokens_groups:
        for word, _ in pos_tag(entry):
            if word not in STOPWORDS:
                result_words.append(word)
    return [word_mapper.get(word, word) for word in result_words]
