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
