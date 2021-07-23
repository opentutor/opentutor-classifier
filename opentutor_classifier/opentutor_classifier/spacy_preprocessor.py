#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from opentutor_classifier.spacy_model import find_or_load_spacy
import string
from os import path

"""
This class contains the methods that operate on the questions to normalize them. The questions are tokenized, punctuations are
removed and words are stemmed to bring them to a common platform
"""


class SpacyPreprocessor(object):
    def __init__(self, shared_root):
        self.punct = set(string.punctuation)
        self.model = find_or_load_spacy(path.join(shared_root, "spacy-model"))

    def inverse_transform(self, x):
        return [" ".join(doc) for doc in x]

    def transform(self, x):
        return list(self.tokenize(x))

    """
    Tokenizes the input question. It also performs case-folding and stems each word in the question using Porter's Stemmer.
    """

    def tokenize(self, sentence):
        doc = self.model(sentence)
        for token in doc:
            if all(char in self.punct for char in token.lemma_):
                continue
            try:
                stemmed_token = token.lemma_
            except BaseException:
                print(
                    "Unicode error. File encoding was changed when you opened it in Excel. ",
                    end=" ",
                )
                print(
                    "This is most probably an error due to csv file from Google docs being opened in Word. ",
                    end=" ",
                )
                print(
                    "Download the file from Google Docs and DO NOT open it in Excel. Run the program immediately. ",
                    end=" ",
                )
                print(
                    "If you want to edit using Excel and then follow instructions at: "
                )
                print(
                    "http://stackoverflow.com/questions/6002256/is-it-possible-to-force-excel-recognize-utf-8-csv-files-automatically"
                )
                continue

            yield stemmed_token
