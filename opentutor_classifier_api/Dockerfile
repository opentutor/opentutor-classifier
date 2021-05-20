FROM python:3.8-slim
ENV STATUS_URL_FORCE_HTTPS=false
WORKDIR /tmp/build
COPY opentutor_classifier opentutor_classifier
RUN pwd
RUN ls -la /tmp/build
ADD opentutor_classifier_api/requirements.txt opentutor_classifier_api/requirements.txt
RUN cd opentutor_classifier_api && pip install -r requirements.txt
WORKDIR /app
RUN rm -rf /tmp/build
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