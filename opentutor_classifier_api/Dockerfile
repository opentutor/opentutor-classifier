FROM python:3.8-slim
ENV STATUS_URL_FORCE_HTTPS=false
WORKDIR /app
RUN pip install poetry
COPY opentutor_classifier ./opentutor_classifier
# build opentutor-classifier as something api can install
RUN cd opentutor_classifier \
	&& make poetry-build-classifier
COPY opentutor_classifier_api ./opentutor_classifier_api
RUN cd opentutor_classifier_api \
	&& poetry config virtualenvs.create false \
	&& poetry install --no-dev
# COPY build /tmp/build
# RUN pip install /tmp/build/opentutor_classifier
# RUN rm -rf /tmp/build
ENV FLASK_APP=opentutor_classifier_api
WORKDIR /app
COPY opentutor_classifier_api/src .
RUN python -m nltk.downloader punkt \
	&& python -m nltk.downloader wordnet \
	&& python -m nltk.downloader averaged_perceptron_tagger \
	&& python -m nltk.downloader stopwords
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]