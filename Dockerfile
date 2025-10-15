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

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# We are actually working with confluent_Kafka version 2.6.1. 
RUN pip install --no-cache-dir \
confluent_Kafka

# Python 3.13 requires this to be compatible with pytorch
RUN pip install --upgrade typing_extensions

# Set the working directory inside the container
WORKDIR /consumer

ARG CACHE_BUST=1

# Clone the repository
RUN git clone https://github.com/DIETI-DISTA-IoT/Train_IoT_local_anomaly_detection.git .

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask API port
EXPOSE 5000

# Command to start the application
ENV PYTHONUNBUFFERED=1
CMD ["python", "consume.py"]