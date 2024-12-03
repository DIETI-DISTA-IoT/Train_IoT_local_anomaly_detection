# Use Debian slim as the base image for a lightweight but more flexible base
FROM python:3.13-slim-bullseye

# Install system dependencies required for building Python packages and PyTorch dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    librdkafka-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /consumer

# Clone the repository
RUN git clone https://github.com/DIETI-DISTA-IoT/Train_IoT_local_anomaly_detection.git .

# Set additional environment variables for Kafka connection
ENV KAFKA_BROKER="kafka:9092"
ENV VEHICLE_NAME=""

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch and other dependencies
# Note: You may need to adjust the PyTorch installation command based on your specific requirements
# This is a generic CPU-only installation - modify as needed
RUN pip install --no-cache-dir -r requirements.txt

# Command to start the application
CMD ["python", "consumer_synthetic_data.py"]