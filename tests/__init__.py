import os


def fixture_path(p: str) -> str:
    return os.path.abspath(os.path.join(".", "tests", "fixtures", p))


def fixture_path_word2vec_model(p2: str) -> str:
    return os.path.abspath(os.path.join(".", "tests", "fixtures", p2))
