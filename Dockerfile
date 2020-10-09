FROM python:3.8-slim
WORKDIR /opt/opentutor_classifier
COPY . .
RUN apt-get update && apt-get install -y git
RUN pip install .
WORKDIR /app
RUN rm -rf /opt/opentutor_classifier
RUN python -m nltk.downloader punkt \
	&& python -m nltk.downloader wordnet \
	&& python -m nltk.downloader averaged_perceptron_tagger \
	&& python -m nltk.downloader stopwords
COPY opentutor_classifier_tasks .
COPY bin/training_worker.sh .
RUN chmod a+x training_worker.sh
ENTRYPOINT ["opentutor_classifier"]
CMD [ "train" ]