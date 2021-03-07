FROM ubuntu:bionic

RUN apt update &&\
    apt -y install libsndfile1 &&\
    apt install -y software-properties-common &&\
    apt install -y python3-pip &&\

COPY requirements.txt /inferenciator/

RUN pip3 install -r /inferenciator/requirements.txt

COPY ./ /inferenciator/

WORKDIR /inferenciator/

CMD ["python3","inferencerStream.py"]
