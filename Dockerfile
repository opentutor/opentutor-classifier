FROM python:3.8-slim
WORKDIR /opt/opentutor_classifier
COPY . .
RUN pip install .
WORKDIR /app
RUN rm -rf /opt/opentutor_classifier
ENTRYPOINT ["opentutor_classifier"]
CMD [ "train" ]
