FROM ubuntu:23.10
RUN apt-get update
RUN apt-get install python3 python3-pip -y
RUN apt install python3-pandas -y && \
    apt install python3-torch -y && \
    apt install python3-torchvision -y && \
    apt install python3-tqdm -y && \
    apt install python3-numpy -y


COPY . .
RUN make . .