import re
from typing import Any, Dict

CAMEL_TO_SNAKE_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")


def rename_camel_to_snake(s: str) -> str:
    return CAMEL_TO_SNAKE_PATTERN.sub("_", s).lower()


def dict_camel_to_snake(d: Dict[str, Any]) -> Dict[str, Any]:
    return {rename_camel_to_snake(k): v for k, v in d.items()}
