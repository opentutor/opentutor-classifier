FROM python:3.8 as builder
WORKDIR /build
RUN pip install poetry
COPY opentutor_classifier ./opentutor_classifier
# build opentutor-classifier as something api can install
RUN apt-get update \
	&& apt-get install -y --no-install-recommends gcc \
	&& apt-get install -y --fix-missing swig libpulse-dev libsdl2-dev libasound2-dev \
	&& rm -rf /var/lib/apt/lists/*
RUN cd opentutor_classifier \
	&& poetry config virtualenvs.create false \
	&& poetry install --no-dev \
	&& poetry build
FROM python:3.8-slim
ENV STATUS_URL_FORCE_HTTPS=false
WORKDIR /app
RUN pip install poetry
WORKDIR /app/opentutor_classifier
# the only thing we need from opentutor_classifier is the built .gz (to install)
COPY --from=builder /build/opentutor_classifier/dist ./dist
WORKDIR /app
COPY opentutor_classifier_api ./opentutor_classifier_api
RUN apt-get update \
	&& apt-get install -y --no-install-recommends gcc \
	&& apt-get install -y --fix-missing swig libpulse-dev libsdl2-dev libasound2-dev \
	&& rm -rf /var/lib/apt/lists/*
RUN cd opentutor_classifier_api \
	&& poetry config virtualenvs.create false \
	&& poetry install --no-dev \
	&& pip uninstall -y poetry
ENV FLASK_APP=opentutor_classifier_api
WORKDIR /app
COPY opentutor_classifier_api/src .
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]