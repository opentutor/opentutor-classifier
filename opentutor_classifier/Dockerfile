FROM python:3.8-slim
WORKDIR /app
COPY poetry.lock pyproject.toml ./
RUN apt-get update \
	&& apt-get install -y --no-install-recommends gcc \
	&& apt-get install -y --fix-missing swig libpulse-dev libsdl2-dev libasound2-dev \
	&& rm -rf /var/lib/apt/lists/*
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