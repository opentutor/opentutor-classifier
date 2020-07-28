FROM python:3.8-slim
WORKDIR /opt/opentutor_classifier
COPY . .
RUN pip install .
WORKDIR /app
RUN rm -rf /opt/opentutor_classifier
RUN python -m nltk.downloader punkt \
	&& python -m nltk.downloader wordnet \
	&& python -m nltk.downloader averaged_perceptron_tagger \
	&& python -m nltk.downloader stopwords
ENTRYPOINT ["opentutor_classifier"]
CMD [ "traindefault", "train"]
