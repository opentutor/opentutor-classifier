#!/bin/bash
##
## This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
## Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
##
## The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
##


if [ -x "$(command -v pyenv)" ]; then
    # if we're using pyenv,
    # make sure the version specified in the .python-version file
    # is installed
    pyenv install --skip-existing
fi


if ! [ -x "$(command -v poetry)" ]; then
    # if poetry is not installed (and we're on linux or mac), then install it
    if ! [ -x "$(command -v curl)" ]; then
        # ...unless you don't have curl, in which case, install poetry yourself
        echo "You need to install poetry to develop on this project."
        echo "https://python-poetry.org/docs/"
        exit 1
    fi
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    . $HOME/.poetry/env
fi
