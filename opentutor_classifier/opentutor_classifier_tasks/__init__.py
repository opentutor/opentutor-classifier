#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(verbose=True, override=True, dotenv_path=os.path.join(os.getcwd(), ".env"))

from celery import signals  # noqa E402
from celery.utils.log import get_task_logger  # noqa E402

dotenv_path = os.path.join(os.getcwd(), ".env")


@signals.after_setup_logger.connect()
def logger_setup_handler(logger: logging.Logger, **kwargs):
    logger.addHandler(logging.StreamHandler(sys.stdout))


@signals.task_retry.connect
@signals.task_failure.connect
@signals.task_revoked.connect
def on_task_failure(**kwargs):
    """Abort transaction on task errors."""
    # celery exceptions will not be published to `sys.excepthook`. therefore we have to create another handler here.
    from traceback import format_tb

    get_task_logger(__name__).error(
        "[task:%s:%s]"
        % (
            kwargs.get("task_id"),
            kwargs["sender"].request.correlation_id,
        )
        + "\n"
        + "".join(format_tb(kwargs.get("traceback", [])))
        + "\n"
        + str(kwargs.get("exception", ""))
    )
