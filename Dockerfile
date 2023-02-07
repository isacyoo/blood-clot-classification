FROM pytorch/pytorch:1.9.1-cuda11.1:cudnn8-runtime

WORKDIR /bloodclot

COPY requirements.txt requirements.txt

RUN pip3 install -r -requirements

COPY . .