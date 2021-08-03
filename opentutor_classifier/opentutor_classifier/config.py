from os import environ
from typing import Final

PROP_TRAIN_QUALITY: Final[str] = "TRAIN_QUALITY"
LABEL_GOOD = "good"
LABEL_BAD = "bad"
LABEL_NEUTRAL = "neutral"
LABEL_UNSPECIFIED = ""


def confidence_threshold_default() -> float:
    return float(environ.get("CONFIDENCE_THRESHOLD_DEFAULT", "0.6"))


def get_train_quality_default() -> int:
    return int(environ.get("TRAIN_QUALITY_DEFAULT", 3))
