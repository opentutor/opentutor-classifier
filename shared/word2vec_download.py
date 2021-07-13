#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
import shutil
from zipfile import ZipFile

from utils import download


def word2vec_download(to_path="installed", replace_existing=True) -> str:
    word2vec_path = os.path.abspath(os.path.join(to_path, "word2vec.bin"))
    if os.path.isfile(word2vec_path) and not replace_existing:
        print(f"already is a file! {word2vec_path}")
        return word2vec_path
    word2vec_zip = os.path.join(to_path, "word2vec.zip")
    download("http://vectors.nlpl.eu/repository/20/6.zip", word2vec_zip)
    with ZipFile(word2vec_zip, "r") as z:
        z.extract("model.bin")
    shutil.move("model.bin", word2vec_path)
    os.remove(word2vec_zip)
    return word2vec_path


if __name__ == "__main__":
    word2vec_download()
