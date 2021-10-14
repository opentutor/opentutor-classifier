#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved. 
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from logging.config import dictConfig

from flask import Flask
from flask_cors import CORS
from .config_default import Config


def create_app():
    dictConfig(
        {
            "version": 1,
            "formatters": {
                "default": {
                    "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
                }
            },
            "handlers": {
                "wsgi": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://flask.logging.wsgi_errors_stream",
                    "formatter": "default",
                }
            },
            "root": {"level": "INFO", "handlers": ["wsgi"]},
        }
    )
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)
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

    return app
