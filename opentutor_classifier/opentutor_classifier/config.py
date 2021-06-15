from os import environ

LABEL_GOOD = "good"
LABEL_BAD = "bad"
LABEL_NEUTRAL = "neutral"
LABEL_UNSPECIFIED = ""


def confidence_threshold_default() -> float:
    return float(environ.get("CONFIDENCE_THRESHOLD_DEFAULT", "0.6"))


def use_length_ratio() -> bool:
    return bool(environ.get("FEATURE_LENGTH_RATIO_ENABLED", "False"))


def use_regex_aggregate() -> bool:
    return bool(environ.get("FEATURE_REGEX_AGGREGATE_ENABLED", "False"))
