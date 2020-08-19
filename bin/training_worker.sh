#!/usr/bin/env bash
celery worker --app opentutor_classifier_tasks.tasks.celery --loglevel=info
