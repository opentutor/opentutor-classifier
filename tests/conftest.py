import os
import pytest
import requests
import shutil
from zipfile import ZipFile
from . import fixture_path


def download(url: str, to_path: str):
    r = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(to_path), exist_ok=True)
    with open(to_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def word2vec_download(to_path="shared", replace_existing=True) -> str:
    word2vec_path = os.path.join(to_path, "word2vec.bin")
    if os.path.isfile(word2vec_path):
        return word2vec_path
    word2vec_zip = os.path.join(to_path, "word2vec.zip")
    download("http://vectors.nlpl.eu/repository/20/6.zip", word2vec_zip)
    with ZipFile(word2vec_zip, "r") as z:
        z.extract("model.bin")
    shutil.move("model.bin", word2vec_path)
    os.remove(word2vec_zip)
    return word2vec_path


@pytest.fixture(autouse=True)
def word2vec() -> str:
    return word2vec_download(
        to_path=fixture_path(os.path.join("shared")), replace_existing=False
    )
