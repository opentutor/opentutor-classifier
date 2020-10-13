FROM python:3.8-slim
RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt
COPY build /tmp/build
RUN pip install /tmp/build/opentutor_classifier
RUN rm -rf /tmp/build
ENV FLASK_APP=opentutor_classifier_api
WORKDIR /app
COPY src .
RUN python -m nltk.downloader punkt \
	&& python -m nltk.downloader wordnet \
	&& python -m nltk.downloader averaged_perceptron_tagger \
	&& python -m nltk.downloader stopwords
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]