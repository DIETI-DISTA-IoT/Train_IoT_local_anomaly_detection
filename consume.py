import argparse
import logging
import os
import threading
import json
import sys
import time
from confluent_kafka import Consumer, KafkaError, KafkaException, SerializingProducer
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka.serialization import StringSerializer

from preprocessing import Buffer
from brain import Brain
from reporting import MetricsReporter


# Configure logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read Kafka configuration from environment variables
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')  # Kafka broker URL
VEHICLE_NAME=os.getenv('VEHICLE_NAME')
CONTAINER_NAME=os.getenv('CONTAINER_NAME')


logging.info(f"Starting consumer system for vehicle: {VEHICLE_NAME}")



received_all_real_msg = 0
received_anomalies_msg = 0
received_normal_msg = 0


def create_consumer():
    # Kafka consumer configuration
    conf_cons = {
        'bootstrap.servers': KAFKA_BROKER,  # Kafka broker URL
        'group.id': f'{VEHICLE_NAME}-consumer-group',  # Consumer group ID for message offset tracking
        'auto.offset.reset': 'earliest'  # Start reading from the earliest message if no offset is present
    }
    return Consumer(conf_cons)


def create_producer_statistic():
    conf_prod_stat={
        'bootstrap.servers': KAFKA_BROKER,  # Kafka broker URL
        'key.serializer': StringSerializer('utf_8'),
        'value.serializer': lambda v, ctx: json.dumps(v)
    }
    return SerializingProducer(conf_prod_stat)


def check_and_create_topics(topic_list):
    """
    Check if the specified topics exist in Kafka, and create them if missing.

    Args:
        topic_list (list): List of topic names to check/create.
    """
    admin_client = AdminClient({'bootstrap.servers': KAFKA_BROKER})
    existing_topics = admin_client.list_topics(timeout=10).topics.keys()

    topics_to_create = [
        NewTopic(topic, num_partitions=1, replication_factor=1)
        for topic in topic_list if topic not in existing_topics
    ]

    if topics_to_create:
        logging.info(f"Creating missing topics: {[topic.topic for topic in topics_to_create]}")
        result = admin_client.create_topics(topics_to_create)

        for topic, future in result.items():
            try:
                future.result()
                logging.info(f"Topic '{topic}' created successfully.")
            except KafkaException as e:
                logging.error(f"Failed to create topic '{topic}': {e}")


def deserialize_message(msg):
    """
    Deserialize the JSON-serialized data received from the Kafka Consumer.

    Args:
        msg (Message): The Kafka message object.

    Returns:
        dict or None: The deserialized Python dictionary if successful, otherwise None.
    """
    try:
        # Decode the message and deserialize it into a Python dictionary
        message_value = json.loads(msg.value().decode('utf-8'))
        logging.info(f"{CONTAINER_NAME}: received message from topic [{msg.topic()}]")
        return message_value
    except json.JSONDecodeError as e:
        logging.error(f"{CONTAINER_NAME}: Error deserializing message: {e}")
        return None


def process_message(topic, msg, producer):
    """
        Process the deserialized message based on its topic.
    """
    global received_all_real_msg, received_anomalies_msg, received_normal_msg

    logging.info(f"Processing message from topic [{topic}]")
    if topic.endswith("_anomalies"):
        logging.debug(f"{CONTAINER_NAME}: ANOMALIES - Processing message")
        anomalies_buffer.add(msg)
        received_anomalies_msg += 1
    elif topic.endswith("_normal_data"):
        logging.debug(f"{CONTAINER_NAME}: NORMAL DATA - Processing message")
        diagnostics_buffer.add(msg)
        received_normal_msg += 1

    received_all_real_msg += 1


def consume_vehicle_data():
    """
        Consume messages for a specific vehicle from Kafka topics.
    """
    topic_anomalies = f"{VEHICLE_NAME}_anomalies"
    topic_normal_data = f"{VEHICLE_NAME}_normal_data"
    topic_statistics= f"{VEHICLE_NAME}statistics"

    check_and_create_topics([topic_anomalies,topic_normal_data, topic_statistics])

    consumer = create_consumer()
    producer = create_producer_statistic()

    consumer.subscribe([topic_anomalies, topic_normal_data])
    logging.info(f"{CONTAINER_NAME} will start consuming {topic_anomalies}, {topic_normal_data}")

    try:
        while True:
            msg = consumer.poll(5.0)  # Poll per 1 secondo
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logging.info(f"{CONTAINER_NAME}: End of partition reached for {msg.topic()}")
                else:
                    logging.error(f"{CONTAINER_NAME}: consumer error: {msg.error()}")
                continue

            deserialized_data = deserialize_message(msg)
            if deserialized_data:
                process_message(msg.topic(), deserialized_data, producer)

    except KeyboardInterrupt:
        logging.info(f"{CONTAINER_NAME}: consumer interrupted by user.")
    except Exception as e:
        logging.error(f"{CONTAINER_NAME}: error in consumer for {VEHICLE_NAME}: {e}")
    finally:
        consumer.close()
        logging.info(f"{CONTAINER_NAME}: consumer for {VEHICLE_NAME} closed.")


def train_model(**kwargs):
    global brain

    batch_size = kwargs.get('batch_size', 32)
    while True:

        anomalies_loss = diagnostics_loss = 0

        if diagnostics_buffer.size >= batch_size and anomalies_buffer.size >= batch_size:
            diagnostics_feats, diagnostics_labels = diagnostics_buffer.sample(batch_size)
            anomalies_feats, anomalies_labels = anomalies_buffer.sample(batch_size)

            if len(diagnostics_feats) > 0:
                diagnostics_loss = brain.train_step(diagnostics_feats, diagnostics_labels)
                diagnostics_loss /= len(diagnostics_feats)

            if len(anomalies_feats) > 0:
                anomalies_loss = brain.train_step(anomalies_feats, anomalies_labels)
                anomalies_loss /= len(anomalies_feats)

        total_loss = anomalies_loss + diagnostics_loss

        if total_loss > 0:
            metrics_reporter.report({
                'anomalies_loss': anomalies_loss, 
                'diagnostics_loss': diagnostics_loss, 
                'total_loss': total_loss})

        time.sleep(kwargs.get('training_freq_seconds', 1))


def main(argv):
    """
        Start the consumer for the specific vehicle.
    """
    global VEHICLE_NAME, CONTAINER_NAME, KAFKA_BROKER
    global batch_size
    global anomalies_buffer, diagnostics_buffer, brain, metrics_reporter

    parser = argparse.ArgumentParser(description='Start the consumer for the specific vehicle.')
    parser.add_argument('--vehicle_name', type=str, required=True, help='Name of the vehicle')
    parser.add_argument('--container_name', type=str, default='generic_consumer', help='Name of the container')
    parser.add_argument('--kafka_broker', type=str, default='kafka:9092', help='Kafka broker URL')
    parser.add_argument('--buffer_size', type=int, default=10, help='Size of the buffer')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of the batch')
    parser.add_argument('--logging_level', type=str, default='INFO', help='Logging level')
    args = parser.parse_args(argv)

    VEHICLE_NAME = args.vehicle_name
    CONTAINER_NAME = args.container_name
    KAFKA_BROKER = args.kafka_broker
    

    brain = Brain(**vars(args))
    metrics_reporter = MetricsReporter(**vars(args))

    anomalies_buffer = Buffer(args.buffer_size, label=1)
    diagnostics_buffer = Buffer(args.buffer_size, label=0)

    thread1=threading.Thread(target=consume_vehicle_data)
    thread1.daemon=True

    training_thread=threading.Thread(target=train_model, kwargs=vars(args))
    training_thread.daemon=True
    
    thread1.start()
    training_thread.start()

    thread1.join()
    training_thread.join()


if __name__=="__main__":
    main(sys.argv[1:])