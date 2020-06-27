import os


def fixture_path(p: str) -> str:
    return os.path.abspath(os.path.join(".", "tests", "fixtures", p))
