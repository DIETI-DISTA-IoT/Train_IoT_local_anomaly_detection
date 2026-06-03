# Use Debian slim as the base image
FROM python:3.13-slim-bullseye

# Install system dependencies required for building Python packages and Kafka dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    cmake \
    pkg-config \
    libssl-dev \
    libsasl2-dev \
    librdkafka-dev \
    librdkafka1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set additional environment variables for Kafka connection
ENV KAFKA_BROKER="kafka:9092"
ENV VEHICLE_NAME=""
ENV CONTAINER_NAME="generic_consumer"

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Full rebuild bust: pass CACHE_BUST=<timestamp> to re-run pip install AND code clone.
# Used by:  make build-consumer-scache
ARG CACHE_BUST=1

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# We are actually working with confluent_Kafka version 2.6.1.
RUN pip install --no-cache-dir \
    confluent_kafka

# Python 3.13 requires this to be compatible with pytorch
RUN pip install --upgrade typing_extensions

# Install dependencies from the build context (submodule checkout on disk).
# This layer is cached when using scache-nolib; re-run only when using scache.
COPY consumer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Code-only bust: pass CODE_BUST=<timestamp> to re-run only the git clones, keeping pip cached.
# Used by:  make build-consumer-scache-nolib
ARG CODE_BUST=1

WORKDIR /consumer

RUN git clone --branch sereBench https://github.com/DIETI-DISTA-IoT/Train_IoT_local_anomaly_detection.git .
RUN git clone --branch sereBench https://github.com/DIETI-DISTA-IoT/of-core OpenFAIR/

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
CMD ["python", "consume.py"]
