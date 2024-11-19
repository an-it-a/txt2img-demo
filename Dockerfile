FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3&& \
    rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

RUN apt-get install wget

RUN mkdir /app/models
WORKDIR /app/models
# beautifulRealistic_v60.safetensors
RUN wget "https://civitai.com/api/download/models/113479?type=Model&format=SafeTensor&size=pruned&fp=fp16" --content-disposition
# vaeFtMse840000EmaPruned_vae.safetensors
RUN wget "https://civitai.com/api/download/models/311162?type=Model&format=SafeTensor" --content-disposition
# ng_deepnegative_v1_75t.pt
RUN wget "https://civitai.com/api/download/models/5637?type=Model&format=PickleTensor&size=full&fp=fp16" --content-disposition
# bad-hands-5.pt
RUN wget "https://civitai.com/api/download/models/125849?type=Model&format=PickleTensor" --content-disposition
WORKDIR /app

COPY main.py /app

CMD exec functions-framework --target=main_handle