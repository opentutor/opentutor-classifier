#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
import tarfile

from utils import download


def spacy_download(to_path="installed", replace_existing=True) -> str:
    spacy_path = os.path.abspath(os.path.join(to_path, "spacy-model"))
    if os.path.isfile(spacy_path) and not replace_existing:
        print(f"already is a file! {spacy_path}")
        return spacy_path
    spacy_tar = os.path.join(to_path, "spacy.tar.gz")
    download(
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0.tar.gz",
        spacy_tar,
    )
    tar = tarfile.open(spacy_tar)
    tar.extractall(spacy_path)
    os.remove(spacy_tar)
    return spacy_path


if __name__ == "__main__":
    spacy_download()
