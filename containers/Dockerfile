FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
MAINTAINER IIT-Jodhpur
ENV PYTHONUNBUFFERED 1
WORKDIR /app

COPY requirement.txt .
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update 
RUN apt-get install -y python3 python3-pip build-essential libssl-dev libffi-dev python3-dev
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx
RUN pip install --upgrade pip && pip install --no-cache -r requirement.txt

RUN ["mkdir", "-p", "inputs"]
COPY . .
RUN ["chmod", "+x", "containers/commands.sh" ]
RUN ["chmod", "+x", "containers/run-dbnet.sh" ]
