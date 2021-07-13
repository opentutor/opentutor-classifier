FROM python:3.8-slim
WORKDIR /opt/opentutor_classifier
COPY . .
# RUN pip install .
WORKDIR /app

COPY poetry.lock pyproject.toml /app/
# ENV PYTHONPATH=${PYTHONPATH}:${PWD} 

# set version of poetry to install?
RUN pip install poetry 
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

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