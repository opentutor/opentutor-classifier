#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import logging
import json
import os
from logging.config import dictConfig

from flask import Flask, has_request_context, request, g
from flask_cors import CORS
from .config_default import Config

if os.environ.get("IS_SENTRY_ENABLED", "") == "true":
    import sentry_sdk  # NOQA E402
    from sentry_sdk.integrations.flask import FlaskIntegration  # NOQA E402


class JSONFormatter(logging.Formatter):
    RECORD_ATTRS = [
        "name",
        "levelname",
        "filename",
        "module",
        "path",
        "lineno",
        "funcName",
        "endpoint",
        "method",
        "url",
        "headers",
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

    def to_json(self, payload):
        # do not assume there's a Flask request context here so must use FLASK_ENV env var not app.debug
        indent = 2 if os.environ.get("FLASK_ENV", "") == "development" else None
        return json.dumps(payload, indent=indent)

    def format(self, record):
        payload = self.to_payload(record)
        return self.to_json(payload)


class RequestJSONFormatter(JSONFormatter):
    def format(self, record):
        if has_request_context():
            # todo for distributed tracing:
            # record.request_id = g.request_id if hasattr(g, 'request_id') else '-'
            if hasattr(g, "response_time"):
                setattr(record, "response-time", g.response_time)
            record.path = request.full_path
            record.endpoint = request.endpoint
            record.method = request.method
            record.url = request.url
            # make sure to redact sensitive info: cookies, auth...
            record.headers = {
                k: v for k, v in request.headers.items() if "auth" not in k.lower()
            }

        return super().format(record)


class RequestFilter(logging.Filter):

    # def __init__(self, methods=None):
    #     self.methods = methods or []
    #     super().__init__()

    def filter(self, record):
        # TODO redact sensitive data
        return True
        # if hasattr(record, 'method'):
        #     if record.method in self.methods:
        #         return True
        # else:
        #     return True


def create_app():
    log_level = os.environ.get("LOG_LEVEL_UPLOAD_API", "INFO")
    log_format = os.environ.get("LOG_FORMAT_UPLOAD_API", "json")
    dictConfig(
        {
            "version": 1,
            "formatters": {
                "default": {"()": "opentutor_classifier_api.JSONFormatter"},
                "simple": {"format": "%(levelname)s %(message)s"},
                "verbose": {
                    "format": "[%(asctime)s] - %(name)s: %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]"
                },
                "json": {"()": "opentutor_classifier_api.JSONFormatter"},
                "request_json": {"()": "opentutor_classifier_api.RequestJSONFormatter"},
            },
            "filters": {"requests": {"()": "opentutor_classifier_api.RequestFilter"}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": log_format,
                    "level": log_level,
                    "stream": "ext://sys.stdout",
                },
                "wsgi": {
                    "class": "logging.StreamHandler",
                    "formatter": log_format,
                    "stream": "ext://flask.logging.wsgi_errors_stream",
                },
                "request": {
                    "class": "logging.StreamHandler",
                    "formatter": "request_json",
                    "filters": ["requests"],
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {"level": log_level, "handlers": ["wsgi"]},
            # stop propagation otherwise root logger will also run
            "loggers": {
                "request": {"level": log_level, "propagate": 0, "handlers": ["request"]}
            },
        }
    )
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    if os.environ.get("IS_SENTRY_ENABLED", "") == "true":
        sentry_sdk.init(
            dsn=os.environ.get("SENTRY_DSN_OPENTUTOR_CLASSIFIER"),
            # include project so issues can be filtered in sentry:
            environment=os.environ.get("PYTHON_ENV", "opentutor-qa"),
            integrations=[FlaskIntegration()],
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            # We recommend adjusting this value in production.
            traces_sample_rate=0.20,
            debug=os.environ.get("SENTRY_DEBUG_CLASSIFIER", "") == "true",
        )

    from opentutor_classifier_api.blueprints.evaluate import eval_blueprint

    app.register_blueprint(eval_blueprint, url_prefix="/classifier/evaluate")

    from opentutor_classifier_api.blueprints.ping import ping_blueprint

    app.register_blueprint(ping_blueprint, url_prefix="/classifier/ping")

    from opentutor_classifier_api.blueprints.train import train_blueprint

    app.register_blueprint(train_blueprint, url_prefix="/classifier/train")

    from opentutor_classifier_api.blueprints.train_default import (
        train_default_blueprint,
    )

    app.register_blueprint(
        train_default_blueprint, url_prefix="/classifier/train_default"
    )

    from opentutor_classifier_api.blueprints.healthcheck import healthcheck_blueprint

    app.register_blueprint(healthcheck_blueprint, url_prefix="/classifier/healthcheck")

    from opentutor_classifier_api.blueprints.training_data import trainingdata_blueprint

    app.register_blueprint(
        trainingdata_blueprint, url_prefix="/classifier/training_data"
    )

    from opentutor_classifier_api.blueprints.training_config import (
        trainingconfig_blueprint,
    )

    app.register_blueprint(
        trainingconfig_blueprint, url_prefix="/classifier/training_config"
    )

    @app.route("/classifier/error")
    def error_handler_test():
        raise Exception("Safe to ignore, route for intentional error")

    return app
