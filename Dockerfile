FROM continuumio/miniconda3:latest

RUN conda create -n images python=3.8 -y
RUN echo "source activate images" > ~/.bashrc
ENV PATH /opt/conda/envs/images/bin:$PATH

RUN pip install torch torchvision && \
    pip install mlflow && \
    pip install numpy && \
    pip install tqdm && \
    pip install torcheval


COPY . .

SHELL ["conda", "run", "--name", "images", "/bin/bash", "-c"]

ENTRYPOINT [ "conda", "run", "--name", "images", "python", "model/train.py" ]