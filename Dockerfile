# 1=When i use this first dockerfile i got an error with pandas installation

#FROM python:3.7.6
#RUN pip install --upgrade pip

#WORKDIR /app
#ENV FLASK_APP=app.py

#COPY requirements.txt .

#RUN pip install -r requirements.txt                   


#COPY . .

#EXPOSE 5000

#CMD ["python", "app.py"]"


#2 i found this internet and it works fine

FROM frolvlad/alpine-python-machinelearning:latest

RUN pip install --upgrade pip

WORKDIR /app

COPY . /app
RUN apk add build-base
RUN apk add --no-cache --virtual .build-deps g++ python3-dev libffi-dev openssl-dev && \
    apk add --no-cache --update python3 && \
    pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader punkt
EXPOSE 5000

ENTRYPOINT  ["python"]

CMD ["app1.py"]
