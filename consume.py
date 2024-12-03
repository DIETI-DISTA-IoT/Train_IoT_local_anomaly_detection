import argparse
import logging
import os
import threading
import json
import time

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer, SerializingProducer
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka.serialization import StringSerializer


# Configure logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read Kafka configuration from environment variables
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')  # Kafka broker URL
VEHICLE_NAME=os.getenv('VEHICLE_NAME')
CONTAINER_NAME=os.getenv('CONTAINER_NAME')

# Validate that KAFKA_BROKER and TOPIC_NAME are set
if not KAFKA_BROKER:
    raise ValueError("Environment variable KAFKA_BROKER is missing.")
if not VEHICLE_NAME:
    raise ValueError("Environment variable VEHICLE_NAME is missing.")

logging.info(f"Starting consumer system for vehicle: {VEHICLE_NAME}")

# List to store received messages and a constant for the maximum number of stored messages
real_msg_list = []  # Stores real sensor messages
anomalies_msg_list = []
normal_msg_list = []

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


def produce_statistics(producer):
    """
    Publish the current statistics to the VEHICLE_NAME_statistics topic.

    Args:
        producer (Producer): The Kafka producer instance.
    """
    global received_all_real_msg, received_anomalies_msg, received_normal_msg
    stats = {
        'vehicle_name' : VEHICLE_NAME,
        'total_messages': received_all_real_msg,
        'anomalies_messages': received_anomalies_msg,
        'normal_messages': received_normal_msg
    }
    topic_statistics=f"{VEHICLE_NAME}_statistics"
    try:
        producer.produce(topic=topic_statistics, value=stats)
        producer.flush()
        logging.info(f"{CONTAINER_NAME}: published to topic: {topic_statistics}")
    except Exception as e:
        logging.error(f"{CONTAINER_NAME}: Failed to produce statistics: {e}")


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
        anomalies_msg_list.append(msg)
        received_anomalies_msg += 1
    elif topic.endswith("_normal_data"):
        logging.debug(f"{CONTAINER_NAME}: NORMAL DATA - Processing message")
        normal_msg_list.append(msg)
        received_normal_msg += 1

    real_msg_list.append(msg)
    received_all_real_msg += 1
    logging.debug(f"{CONTAINER_NAME}: DATA ({topic}) - {msg}")

    produce_statistics(producer)


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


def main():
    """
        Start the consumer for the specific vehicle.
    """
    thread1=threading.Thread(target=consume_vehicle_data)
    thread1.daemon=True
    
    thread1.start()
    thread1.join()


if __name__=="__main__":
    main()