
from serverless_modules.train_job import QuestionConfig, ExpectationConfig
import yaml


from typing import Optional
import pandas as pd


# TODO this should never return None, but code currently depends on that
def dict_to_config(config_data: dict) -> Optional[QuestionConfig]:
    return (
        QuestionConfig(
            question=config_data.get("question", ""),
            expectations=[
                ExpectationConfig(ideal=i["ideal"])
                for i in config_data.get("expectations", [])
            ],
        )
        if config_data
        else None
    )


def load_config(config_file: str) -> QuestionConfig:
    with open(config_file) as f:
        return QuestionConfig(**yaml.load(f, Loader=yaml.FullLoader))

def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, encoding="latin-1", dtype={"exp_num": str})
