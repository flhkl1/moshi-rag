FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04


RUN apt-get update && apt-get install -y pkg-config libssl-dev curl cmake git libsentencepiece-dev

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /app
ENV AWS_LC_SYS_CMAKE_BUILDER=1
ENV SENTENCEPIECE_NO_PKG_CONFIG=1

WORKDIR /app

ENV EXPORT_DIR=/models
ENV RUST_BACKTRACE=full

CMD ["cargo", "run", "--features", "cuda", "--bin", "moshi-backend", "-r", "--", "--config", "./moshi-backend/config.json", "standalone"]
#CMD ["bash", "-c", "sleep infinity"]