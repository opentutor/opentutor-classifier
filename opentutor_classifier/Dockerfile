FROM python:3.8-slim
WORKDIR /app
COPY poetry.lock pyproject.toml ./
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev
COPY opentutor_classifier ./opentutor_classifier
COPY opentutor_classifier_tasks ./opentutor_classifier_tasks
COPY opentutor_classifier_tasks .
COPY bin/training_worker.sh .
RUN chmod a+x training_worker.sh
ENTRYPOINT ["opentutor_classifier"]
CMD [ "train" ]