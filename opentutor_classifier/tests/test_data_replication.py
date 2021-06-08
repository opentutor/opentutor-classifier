# from os import path
# from typing import List

# import pytest
# import responses

# from opentutor_classifier import (
#     ExpectationTrainingResult,
#     ARCH_SVM_CLASSIFIER,
#     ARCH_LR_CLASSIFIER,
# )
# from opentutor_classifier.config import confidence_threshold_default
# from .utils import (
#     assert_testset_accuracy,
#     assert_train_expectation_results,
#     create_and_test_classifier,
#     fixture_path,
#     read_example_testset,
#     test_env_isolated,
#     train_classifier,
#     train_default_classifier,
#     _TestExpectation,
# )

# CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


# @pytest.fixture(scope="module")
# def data_root() -> str:
#     return fixture_path("data")


# @pytest.fixture(scope="module")
# def shared_root(word2vec) -> str:
#     return path.dirname(word2vec)


# def _test_data_replication(
#     lesson: str,
#     arch: str,
#     # confidence_threshold for now determines whether an answer
#     # is really classified as GOOD/BAD (confidence >= threshold)
#     # or whether it is interpretted as NEUTRAL (confidence < threshold)
#     confidence_threshold: float,
#     expected_training_result: List[ExpectationTrainingResult],
#     expected_accuracy: float,
#     tmpdir,
#     data_root: str,
#     shared_root: str,
# ):
#     with test_env_isolated(
#         tmpdir, data_root, shared_root, arch=arch, lesson=lesson
#     ) as test_config:#_TestConfig
#         rep_factor = [1, 2, 5, 10]
#         data=load_data(self._get_data_file(lesson, _TRAINING_CSV))
#         train_config=TrainingConfig(shared_root=test_config.shared_root)
#         dao=FileDataDao(droot, model_root=train_config.output_dir), #TrainingConfig
#         for i in rep_factor:
#             data_list = [data] * i 
#             new_data = data.concat(data_list)
#             input = TrainingInput(
#                 lesson=lesson,
#                 config=self.find_training_config(lesson), #QuestionConfig
#                 data=new_data, #dataframe
#             )
#             fac = ClassifierFactory()
#             training = fac.new_training(train_config or TrainingConfig(), arch=arch) #TrainingConfig
#             train_result =  training.train(data, dao)

