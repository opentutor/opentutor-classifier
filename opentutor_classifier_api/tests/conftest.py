#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from flask import Response
from src.opentutor_classifier_api import create_app  # type: ignore
from os import path
import pytest


@pytest.fixture
def app():
    myapp = create_app()
    myapp.debug = True
    myapp.response_class = Response
    return myapp


@pytest.fixture(scope="module", autouse=True)
def word2vec() -> str:
    word2vec_path = path.abspath(path.join("..", "shared", "installed", "word2vec.bin"))
    if not path.exists(word2vec_path):
        raise Exception(
            "missing required word2vec.bin. Try run `make clean word2vec.bin` from <PROJECT_ROOT>/word2vec folder"
        )
    return word2vec_path
