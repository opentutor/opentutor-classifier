#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os
import responses
from os import path
from opentutor_classifier.sync import sync
from . import fixture_path


def __sync(tmpdir, lesson: str, url: str):
    output_dir = os.path.join(tmpdir.mkdir("test"), lesson)
    sync(lesson, url, output_dir)
    return output_dir


@responses.activate
def test_sync_data_from_api(tmpdir):
    responses.add(
        responses.POST,
        "https://dev-opentutor.pal3.org/grading-api",
        json={
            "data": {
                "trainingData": {
                    "config": 'question: "What are the challenges to demonstrating integrity in a group?"',
                    "training": "exp_num,text,label\n0,peer pressure,Good",
                }
            }
        },
        status=200,
    )
    output_dir = __sync(tmpdir, "q1", "https://dev-opentutor.pal3.org/grading-api")
    assert path.exists(path.join(output_dir, "training.csv"))
    with open(path.join(output_dir, "training.csv")) as f:
        assert f.read() == "exp_num,text,label\n0,peer pressure,Good"
    assert path.exists(path.join(output_dir, "config.yaml"))
    with open(path.join(output_dir, "config.yaml")) as f:
        assert (
            f.read()
            == 'question: "What are the challenges to demonstrating integrity in a group?"'
        )


def test_sync_data_from_file(tmpdir):
    output_dir = __sync(
        tmpdir, "q1", fixture_path(os.path.join("graphql", "example-1.json"))
    )
    assert path.exists(path.join(output_dir, "training.csv"))
    with open(path.join(output_dir, "training.csv")) as f:
        assert f.read() == "exp_num,text,label\n0,peer pressure,Good"
    assert path.exists(path.join(output_dir, "config.yaml"))
    with open(path.join(output_dir, "config.yaml")) as f:
        assert (
            f.read()
            == 'question: "What are the challenges to demonstrating integrity in a group?"'
        )
