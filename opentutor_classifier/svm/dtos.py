#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from dataclasses import dataclass, asdict
from typing import Dict, List
import yaml

from sklearn import svm


@dataclass
class InstanceExpectationFeatures:
    ideal: List[str]
    good_regex: List[str]
    bad_regex: List[str]


@dataclass
class InstanceConfig:
    question: str
    expectation_features: List[InstanceExpectationFeatures]

    def __post_init__(self):
        self.expectation_features = [
            x
            if isinstance(x, InstanceExpectationFeatures)
            else InstanceExpectationFeatures(**x)
            for x in self.expectation_features
        ]

    def write_to(self, file_path: str):
        with open(file_path, "w") as config_file:
            yaml.safe_dump(asdict(self), config_file)


@dataclass
class InstanceModels:
    models_by_expectation_num: Dict[int, svm.SVC]
    config: InstanceConfig
