#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
from zipfile import ZipFile

from utils import download

DEFAULT_TO_PATH = os.path.join("installed", "sentence-transformer")


def transformer_download(to_path=DEFAULT_TO_PATH, replace_existing=False) -> str:
    transformer_path = os.path.abspath(
        os.path.join(to_path, "distilbert-base-nli-mean-tokens")
    )
    if os.path.exists(transformer_path) and not replace_existing:
        print(f"already downloaded! {transformer_path}")
        return transformer_path
    transformer_zip = os.path.join(to_path, "distilbert-base-nli-mean-tokens.zip")
    download(
        "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/distilbert-base-nli-mean-tokens.zip",
        transformer_zip,
    )
    with ZipFile(transformer_zip, "r") as z:
        z.extractall(transformer_path)
    os.remove(transformer_zip)
    return transformer_path


if __name__ == "__main__":
    transformer_download()
