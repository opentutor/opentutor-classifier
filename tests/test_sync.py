import os
from os import path
from opentutor_classifier.sync import sync, sync_config, sync_training

# import pytest
# from . import fixture_path
import responses


# @pytest.fixture(scope="module")
# def data_root() -> str:
#     return fixture_path("data")


def __sync(tmpdir, lesson: str, url: str):
    output_dir = os.path.join(tmpdir.mkdir("test"), lesson)
    sync(lesson, url, output_dir)
    return output_dir


def __sync_training(tmpdir, lesson: str, url: str):
    output_dir = os.path.join(tmpdir.mkdir("test"), lesson)
    sync_training(lesson, url, output_dir)
    return output_dir


def __sync_config(tmpdir, lesson: str, url: str):
    output_dir = os.path.join(tmpdir.mkdir("test"), lesson)
    sync_config(lesson, url, output_dir)
    return output_dir


@responses.activate
def test_syncs_training_data_for_q1(tmpdir):
    responses.add(
        responses.POST,
        "https://dev-opentutor.pal3.org/grading-api",
        json={
            "data": {"lessonTrainingData": "exp_num,text,label\n0,peer pressure,Good"}
        },
        status=200,
    )
    output_dir = __sync_training(
        tmpdir, "q1", "https://dev-opentutor.pal3.org/grading-api"
    )
    assert path.exists(path.join(output_dir, "training.csv"))
    with open(path.join(output_dir, "training.csv")) as f:
        assert f.read() == "exp_num,text,label\n0,peer pressure,Good\n"


@responses.activate
def test_syncs_config_data_for_q1(tmpdir):
    responses.add(
        responses.POST,
        "https://dev-opentutor.pal3.org/grading-api",
        json={
            "data": {
                "lesson": {
                    "question": "What are the challenges to demonstrating integrity in a group?"
                }
            }
        },
        status=200,
    )
    output_dir = __sync_config(
        tmpdir, "q1", "https://dev-opentutor.pal3.org/grading-api"
    )
    assert path.exists(path.join(output_dir, "config.yaml"))
    with open(path.join(output_dir, "config.yaml")) as f:
        assert (
            f.read()
            == 'question: "What are the challenges to demonstrating integrity in a group?"'
        )
