#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import logging
from logging.config import dictConfig
import os
import json


class JSONFormatter(logging.Formatter):
    RECORD_ATTRS = [
        "name",
        "levelname",
        "filename",
        "path",
        "lineno",
        "funcName",
    ]

    def to_payload(self, record):
        payload = {
            attr: getattr(record, attr)
            for attr in self.RECORD_ATTRS
            if hasattr(record, attr)
        }
        # make sure log messages are consistent across services:
        payload["level"] = payload["levelname"].lower()
        del payload["levelname"]
        payload["logger"] = payload["name"]
        del payload["name"]
        payload["message"] = record.getMessage()
        return payload

    def format(self, record):
        payload = self.to_payload(record)
        return json.dumps(payload)


log_level = os.environ.get("LOG_LEVEL", "DEBUG")
log_format = os.environ.get("LOG_FORMAT", "json")

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {"()": JSONFormatter},
            "simple": {"format": "%(levelname)s %(message)s"},
            "verbose": {
                "format": "[%(asctime)s] - %(name)s: %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]"
            },
            "json": {"()": JSONFormatter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": log_format,
                "level": log_level,
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"level": log_level, "handlers": ["console"]},
    }
)


def get_logger(name="root"):
    return logging.getLogger(name)
