FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 wget&& \
    rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

RUN mkdir /app/models && \
    mkdir /app/models/ckpts && \
    mkdir /app/models/vae && \
    mkdir /app/models/embeddings && \
    mkdir /app/models/loras

#### Download the models if not using Google Cloud Storage
#WORKDIR /app/models/ckpts
## beautifulRealistic_v60.safetensors
#RUN wget "https://civitai.com/api/download/models/113479?type=Model&format=SafeTensor&size=pruned&fp=fp16" --content-disposition
#
#WORKDIR /app/models/vae
## vaeFtMse840000EmaPruned_vae.safetensors
#RUN wget "https://civitai.com/api/download/models/311162?type=Model&format=SafeTensor" --content-disposition
#
#WORKDIR /app/models/embeddings
## easynegative.pt
#RUN wget "https://civitai.com/api/download/models/9536?type=Model&format=PickleTensor&size=full&fp=fp16" --content-disposition
## bad-hands-5.pt
#RUN wget "https://civitai.com/api/download/models/125849?type=Model&format=PickleTensor" --content-disposition
#
#WORKDIR /app/models/loras
## more_details.safetensors
#RUN wget "https://civitai.com/api/download/models/87153?type=Model&format=SafeTensor" --content-disposition

WORKDIR /app

COPY main.py v1-inference.yaml /app/

CMD exec functions-framework --target=main_handle