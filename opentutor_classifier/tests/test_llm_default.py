from os import path
from typing import List
import csv

import pytest
import responses
import uuid
import time

RUN_HEADERS = ['RunID', 'BatchID','Model', 'Temp', 'Batch_Size', 'Run_Time',
                'Max_Tok', 'Prompt_Stem', 'Dialog', 'N_shots', 'C_1', 'C_2',
                    'C_3','C_4','C_5','C_6','Holistic Acc','Avg hit Conf', 'Std Dev Conf',
                    'Precision', 'Recall', 'F1', 'Notes']

from opentutor_classifier import (
    ExpectationTrainingResult,
    ARCH_LR2_CLASSIFIER,
)
#from tests.test_train_classifier import  _test_train_and_predict
from opentutor_classifier.config import confidence_threshold_default
from opentutor_classifier.lr2.constants import MODEL_FILE_NAME
from .utils import (
    fixture_path,
    read_example_testset,
    test_env_isolated,
    train_classifier_N_shot,
    train_default_data_root,
    train_default_classifier,
    show_robust_metrics,
    build_new_result_sheets,
    write_csv,
    save_result
)

def write_txt(target, filename): # Expects a dictionary. 
    f = open('{}.txt'.format(filename), 'w')
    for k, v in target.items():
        if k == " ": f.write(v)
        else:
            f.write(str(k))
            f.write(':')
            f.write(str(v))
            f.write('\n\n')
    f.close()


def write_csv(data, headers, filename): #To build the run csvs and the opentutor csvs
    with open('{}.csv'.format(filename), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(data)

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()

@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)

#@pytest.mark.only
@pytest.mark.parametrize(
    "lesson,arch,",
    [
        (
            "harker_s2",
            ARCH_LR2_CLASSIFIER,
        ),
    ],
)
def test_llm_ot(
    lesson: str,
    arch: str,
    tmpdir,
    data_root: str,
    shared_root: str,
): 
    start = time.time()
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson, is_default_model=True
    ) as test_config:
        mymodel = train_default_classifier(test_config)
        testset = read_example_testset(lesson, confidence_threshold=CONFIDENCE_THRESHOLD_DEFAULT)
        #print("This is what testset looks like", testset)
    
        result = show_robust_metrics(
                arch,
                mymodel.models,
                shared_root,
                testset,
            )
        end = time.time()

        run_report = {x:"NA" for  x in RUN_HEADERS}
        run_report['RunID'] = str(uuid.uuid4())
        run_report['Dialog'] = lesson
        run_report['BatchID'] = 'f58713ed-837d-419a-bd2e-6891cacfae7e'
        run_report['Model'] = 'OpenTutorLogisticRegression'
        run_report['Batch_Size'] = result[3] / result[4]
        run_report['Run_Time'] = int(end) - int(start)
        run_report['N_shots'] = 0
        run_report['Prompt_Stem'] = 'config.yaml'
        run_report['Holistic Acc'] = result[0]
        run_report['Precision'] = result[1][0]
        run_report['Recall'] = result[1][1]
        run_report['F1'] = result[1][2]

        for k,v in result[2].items():
            run_report[k] = v
        #If you need to build a new results sheet. I won't make it dynamic right now.
        #build_new_result_sheets(RUN_HEADERS, 'OT_LLM_Comparison')
        print("This is just to confirm that everything works", run_report)
        run_report = [v for v in run_report.values()]
        save_result(run_report, 'OT_LLM_Comparison')

    assert 1
