from os import environ

LABEL_GOOD = "good"
LABEL_BAD = "bad"
LABEL_NEUTRAL = "neutral"
LABEL_UNSPECIFIED = ""


def confidence_threshold_default() -> float:
    return float(environ.get("CONFIDENCE_THRESHOLD_DEFAULT", "0.6"))

def use_length_ratio() -> bool:
    return bool(environ.get("USE_LENGTH_RATIO", "False"))