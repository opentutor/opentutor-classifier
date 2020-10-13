#!/usr/bin/env bash
celery --app opentutor_classifier_tasks.tasks.celery worker --loglevel=INFO
