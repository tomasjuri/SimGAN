FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter 

WORKDIR /usr/src/app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV TZ=Europe/Prague
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxtst6 \
    libfontconfig1 \
    libxrender1 \
    libsm6 \
    git \
    openssh-client \
    mc \
    vim \
    sudo \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install pipenv
RUN pipenv install
