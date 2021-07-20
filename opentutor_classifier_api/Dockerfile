FROM python:3.8 as builder
WORKDIR /build
RUN pip install poetry
COPY opentutor_classifier ./opentutor_classifier
# build opentutor-classifier as something api can install
RUN cd opentutor_classifier \
	&& poetry config virtualenvs.create false \
	&& poetry install --no-dev \
	&& python -m nltk.downloader punkt \
	&& python -m nltk.downloader wordnet \
	&& python -m nltk.downloader averaged_perceptron_tagger \
	&& python -m nltk.downloader stopwords \
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
RUN cd opentutor_classifier_api \
	&& poetry config virtualenvs.create false \
	&& poetry install --no-dev \
	&& pip uninstall -y poetry
RUN python -m nltk.downloader punkt \
	&& python -m nltk.downloader wordnet \
	&& python -m nltk.downloader averaged_perceptron_tagger \
	&& python -m nltk.downloader stopwords
ENV FLASK_APP=opentutor_classifier_api
WORKDIR /app
COPY opentutor_classifier_api/src .
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]