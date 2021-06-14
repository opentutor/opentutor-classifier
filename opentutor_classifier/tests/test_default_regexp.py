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


# @pytest.mark.parametrize(
#     "example,arch,confidence_threshold,expected_accuracy",
#     [
#         (
#             "proportion",
#             ARCH_LR_CLASSIFIER,
#             CONFIDENCE_THRESHOLD_DEFAULT,
#             1.0,
#         ),
#     ]
# )
# @pytest.mark.only
# def test_train_default(
#     arch: str,
#     data_root: str,
#     shared_root: str,
#     tmpdir,
#     example:str,
#     confidence_threshold: float,
#     expected_accuracy: float,

# ):
#     with test_env_isolated(
#         tmpdir,
#         data_root,
#         shared_root,
#         arch=arch,
#         is_default_model=True,
#         lesson= "default",
#     ) as config:
#         train_result = train_default_classifier(config=config)
#         testset = read_example_testset(
#             example, confidence_threshold=confidence_threshold
#         )
#         assert_testset_accuracy(
#             arch,
#             train_result.models,
#             shared_root,
#             testset,
#             expected_accuracy=expected_accuracy,
#         )


#     def train_default_classifier(config: _TestConfig) -> TrainingResult:
# return train_default_data_root(
#     data_root=path.join(config.data_root, "default"),
#     config=TrainingConfig(shared_root=config.shared_root),
#     output_dir=config.output_dir,
#     arch=config.arch,

#     def train_default_data_root(
#     arch="", config: TrainingConfig = None, data_root="data", output_dir=""
# ) -> TrainingResult:
#     droot, __default__ = path.split(path.abspath(data_root))
#     return train_default(
#         arch=arch, config=config, dao=FileDataDao(droot, model_root=output_dir)
#     )

# def assert_testset_accuracy(
#     arch: str,
#     model_path: str,
#     shared_root: str,
#     testset: _TestSet,
#     expected_accuracy=1.0,
# ) -> None:
#     result = run_classifier_testset(arch, model_path, shared_root, testset)
#     metrics = result.metrics()
#     if metrics.accuracy >= expected_accuracy:
#         return
#     logging.warning("ERRORS:\n" + "\n".join(ex.errors() for ex in result.results))
#     assert metrics.accuracy >= expected_accuracy

# def run_classifier_testset(
#     arch: str, model_path: str, shared_root: str, testset: _TestSet
# ) -> _TestSetResult:
#     model_root, model_name = path.split(model_path)
#     classifier = ClassifierFactory().new_classifier(
#         ClassifierConfig(
#             dao=opentutor_classifier.dao.find_data_dao(),
#             model_name=model_name,
#             model_roots=[model_root],
#             shared_root=shared_root,
#         ),
#         arch=arch,
#     )
#     result = _TestSetResult(testset=testset)
#     for ex in testset.examples:
#         result.results.append(to_example_result(ex, classifier.evaluate(ex.input)))
#     return result

#  def model_and_config(self) -> ModelAndConfig:
# if not self._model_and_config:
#     cm = find_predicton_config_and_pickle(
#         ModelRef(
#             arch=ARCH_LR_CLASSIFIER,
#             lesson=self.model_name,
#             filename=MODEL_FILE_NAME,
#         ),
#         self.dao,
#     )
#     self._model_and_config = (cm.model, cm.config)
# return self._model_and_config
