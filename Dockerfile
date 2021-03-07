FROM python:3.8.8-slim-buster

COPY requirements.txt /inferenciator/

RUN pip3 install -r /inferenciator/requirements.txt

COPY ./ /inferenciator/

WORKDIR /inferenciator/

CMD ["python3","inferencerStream.py"]
