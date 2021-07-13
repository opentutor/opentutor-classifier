#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
#
import os
import requests
from tqdm import tqdm


def download(url: str, to_path: str):
    print("downloading")
    print(f"\tfrom {url}")
    print(f"\tto {to_path}")
    r = requests.get(url, stream=True)
    total_size_in_bytes = int(r.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    os.makedirs(os.path.dirname(to_path), exist_ok=True)
    with open(to_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=block_size):
            if chunk:
                progress_bar.update(len(chunk))
                f.write(chunk)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception("failed to fully download")
