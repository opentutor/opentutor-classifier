#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import os

from celery import Celery

config = {
    "CELERY_BROKER_URL": os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),
    "CELERY_RESULT_BACKEND": os.environ.get(
        "CELERY_RESULT_BACKEND", "redis://redis:6379/0"
    ),
    "CELERY_ACCEPT_CONTENT": ["json"],
    "CELERY_TASK_SERIALIZER": os.environ.get(
        "CELERYCELERY_TASK_SERIALIZER_RESULT_BACKEND", "json"
    ),
    "CELERY_EVENT_SERIALIZER": os.environ.get("CELERY_EVENT_SERIALIZER", "json"),
    "CELERY_RESULT_SERIALIZER": os.environ.get("CELERY_RESULT_SERIALIZER", "json"),
}
celery = Celery("opentutor-classifier-tasks", broker=config["CELERY_BROKER_URL"])
celery.conf.update(config)


@celery.task()
def train_task(lesson):
    pass
