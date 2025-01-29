import argparse
import logging
import threading
import json
import time
from confluent_kafka import Consumer, KafkaError, KafkaException, SerializingProducer
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka.serialization import StringSerializer

from preprocessing import Buffer
from brain import Brain
from communication import MetricsReporter, WeightsReporter, WeightsPuller
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import signal

received_all_real_msg = 0
received_anomalies_msg = 0
received_normal_msg = 0
anomalies_processed = 0
diagnostics_processed = 0
diagnostics_clusters_count = torch.zeros(15)
anomalies_clusters_count = torch.zeros(19)
diagnostics_cluster_percentages =torch.zeros(15)
anomalies_cluster_percentages = torch.zeros(19)


def create_consumer():
    # Kafka consumer configuration
    conf_cons = {
        'bootstrap.servers': KAFKA_BROKER,  # Kafka broker URL
        'group.id': f'{VEHICLE_NAME}-consumer-group',  # Consumer group ID for message offset tracking
        'auto.offset.reset': 'earliest'  # Start reading from the earliest message if no offset is present
    }
    return Consumer(conf_cons)


def get_statistics_producer():
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
        logging.debug(f"received message from topic [{msg.topic()}]")
        return message_value
    except json.JSONDecodeError as e:
        logging.error(f"Error deserializing message: {e}")
        return None


def process_message(topic, msg, producer):
    """
        Process the deserialized message based on its topic.
    """
    global received_all_real_msg, received_anomalies_msg, received_normal_msg

    # logger.debug(f"Processing message from topic [{topic}]")
    if topic.endswith("_anomalies"):
        anomalies_buffer.add(msg)
        received_anomalies_msg += 1
    elif topic.endswith("_normal_data"):
        diagnostics_buffer.add(msg)
        received_normal_msg += 1

    received_all_real_msg += 1


def subscribe_to_topics():
    """
        Subscribe to a list of Kafka topics.
    """
    global consumer

    topics = [f"{VEHICLE_NAME}_anomalies", f"{VEHICLE_NAME}_normal_data"]
    consumer.subscribe(topics)
    global_weights_puller.subscribe()
    logger.debug(f"(re)subscribed to topics: {topics}")


def consume_vehicle_data():
    """
        Consume messages for a specific vehicle from Kafka topics.
    """
    global consumer

    anomaly_topic = f"{VEHICLE_NAME}_anomalies"
    diagnostic_topic = f"{VEHICLE_NAME}_normal_data"
    stats_topic= f"{VEHICLE_NAME}_statistics"
    weights_topic = f"{VEHICLE_NAME}_weights"

    check_and_create_topics([stats_topic, weights_topic])

    consumer = create_consumer()
    producer = get_statistics_producer()

    subscribe_to_topics()

    try:
        while not stop_threads:
            msg = consumer.poll(5.0)  # Poll per 1 secondo
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f"End of partition reached for {msg.topic()}")
                else:
                    logger.error(f"consumer error: {msg.error()}")
                continue

            deserialized_data = deserialize_message(msg)
            if deserialized_data:
                process_message(msg.topic(), deserialized_data, producer)

    except KeyboardInterrupt:
        logger.info(f"consumer interrupted by user.")
    except Exception as e:
        logger.error(f" error in consumer for {VEHICLE_NAME}: {e}")
    finally:
        consumer.close()
        logger.info(f"consumer for {VEHICLE_NAME} closed.")


def push_weights(**kwargs):
    while not stop_threads:
        time.sleep(kwargs.get('weights_push_freq_seconds', 300))
        weights_reporter.push_weights(brain.model.state_dict())


def pull_weights(**kwargs):
    global brain

    while not stop_threads:
        time.sleep(kwargs.get('weights_pull_freq_seconds', 300))        
        new_weights = global_weights_puller.pull_weights()
        if new_weights:
            brain.update_weights(new_weights)
            logger.info("Local weights updated using global model.")


def train_model(**kwargs):
    global brain, diagnostics_processed, anomalies_processed
    global diagnostics_clusters_count, anomalies_clusters_count, diagnostics_cluster_percentages, anomalies_cluster_percentages

    batch_size = kwargs.get('batch_size', 32)
    

    while not stop_threads:
        batch_labels = None
        batch_preds = None
        train_step_done = False
        anomalies_loss = diagnostics_loss = 0

        diagnostics_feats, diagnostics_labels, diagnostics_clusters = diagnostics_buffer.sample(batch_size)
        anomalies_feats, anomalies_labels, anomalies_clusters = anomalies_buffer.sample(batch_size)

        if len(diagnostics_feats) > 0:
            diagnostics_preds, diagnostics_loss = brain.train_step(diagnostics_feats, diagnostics_labels)
            train_step_done = True
            batch_labels = diagnostics_labels
            batch_preds = diagnostics_preds
            diagnostics_processed += len(diagnostics_feats)

        if len(anomalies_feats) > 0:
            anomalies_preds, anomalies_loss = brain.train_step(anomalies_feats, anomalies_labels)
            train_step_done = True
            batch_labels = (anomalies_labels if batch_labels is None else torch.vstack((batch_labels, anomalies_labels)))
            batch_preds = (anomalies_preds if batch_preds is None else torch.vstack((batch_preds, anomalies_preds)))
            anomalies_processed += len(anomalies_feats)


        if train_step_done:
            
            # convert bath_preds to binary using pytorch:
            batch_preds = (batch_preds > 0.5).float()
            total_loss = anomalies_loss + diagnostics_loss
            batch_accuracy = accuracy_score(batch_labels, batch_preds)
            batch_precision = precision_score(batch_labels, batch_preds, zero_division=0)
            batch_recall = recall_score(batch_labels, batch_preds)
            batch_f1 = f1_score(batch_labels, batch_preds)

            if len(diagnostics_clusters) > 0:
                batch_diag_clusters = torch.bincount(diagnostics_clusters.squeeze(-1), minlength=15)
                diagnostics_clusters_count += batch_diag_clusters
                diagnostics_cluster_percentages = diagnostics_clusters_count / diagnostics_processed
                        
            if len(anomalies_clusters) > 0:
                batch_anom_clusters = torch.bincount(anomalies_clusters.squeeze(-1), minlength=19)
                anomalies_clusters_count += batch_anom_clusters
                anomalies_cluster_percentages = anomalies_clusters_count / anomalies_processed


            metrics_reporter.report({
                'anomalies_loss': anomalies_loss, 
                'diagnostics_loss': diagnostics_loss, 
                'total_loss': total_loss,
                'accuracy': batch_accuracy,
                'precision': batch_precision,
                'recall': batch_recall,
                'f1': batch_f1,
                'diagnostics_processed': diagnostics_processed,
                'anomalies_processed': anomalies_processed,
                'diagnostics_cluster_percentages': diagnostics_cluster_percentages.tolist(),
                'anomalies_cluster_percentages': anomalies_cluster_percentages.tolist()})

        time.sleep(kwargs.get('training_freq_seconds', 1))


def signal_handler(sig, frame):
    global stop_threads, stats_consuming_thread, training_thread, pushing_weights_thread, pulling_weights_thread
    logger.debug(f"Received signal {sig}. Gracefully stopping {VEHICLE_NAME} producer.")
    stop_threads = True


def resubscribe():
    while  not stop_threads:
        try:
            # Wait for a certain interval before resubscribing
            time.sleep(resubscribe_interval_seconds)
            subscribe_to_topics()
        except Exception as e:
            logger.error(f"Error in periodic resubscription: {e}")


def main():
    """
        Start the consumer for the specific vehicle.
    """
    global VEHICLE_NAME, KAFKA_BROKER
    global batch_size, stop_threads, stats_consuming_thread, training_thread, pushing_weights_thread, pulling_weights_thread
    global anomalies_buffer, diagnostics_buffer, brain, metrics_reporter, logger, weights_reporter, global_weights_puller
    global resubscribe_interval_seconds

    parser = argparse.ArgumentParser(description='Start the consumer for the specific vehicle.')
    parser.add_argument('--vehicle_name', type=str, required=True, help='Name of the vehicle')
    parser.add_argument('--container_name', type=str, default='generic_consumer', help='Name of the container')
    parser.add_argument('--kafka_broker', type=str, default='kafka:9092', help='Kafka broker URL')
    parser.add_argument('--buffer_size', type=int, default=100, help='Size of the message buffer')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of the batch')
    parser.add_argument('--logging_level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--weights_push_freq_seconds', type=int, default=300, help='Seconds interval between weights push')
    parser.add_argument('--weights_pull_freq_seconds', type=int, default=300, help='Seconds interval between weights pulling from coordinator')
    parser.add_argument('--kafka_topic_update_interval_secs', type=int, default=15, help='Seconds interval between Kafka topic update')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    args = parser.parse_args()

    logger = logging.getLogger(args.container_name)
    logger.setLevel(str(args.logging_level).upper())

    VEHICLE_NAME = args.vehicle_name
    KAFKA_BROKER = args.kafka_broker

    print(f"Starting consumer for vehicle {VEHICLE_NAME}")    

    brain = Brain(**vars(args))
    metrics_reporter = MetricsReporter(**vars(args))
    weights_reporter = WeightsReporter(**vars(args))
    global_weights_puller = WeightsPuller(**vars(args))

    anomalies_buffer = Buffer(args.buffer_size, label=1)
    diagnostics_buffer = Buffer(args.buffer_size, label=0)

    resubscribe_interval_seconds = args.kafka_topic_update_interval_secs
    resubscription_thread = threading.Thread(target=resubscribe)
    resubscription_thread.daemon = True
    
    stats_consuming_thread=threading.Thread(target=consume_vehicle_data)
    stats_consuming_thread.daemon=True

    training_thread=threading.Thread(target=train_model, kwargs=vars(args))
    training_thread.daemon=True

    pushing_weights_thread=threading.Thread(target=push_weights, kwargs=vars(args))
    pushing_weights_thread.daemon=True

    pulling_weights_thread=threading.Thread(target=pull_weights, kwargs=vars(args))
    pulling_weights_thread.daemon=True
    
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))
    stop_threads = False

    stats_consuming_thread.start()
    training_thread.start()
    pushing_weights_thread.start()
    pulling_weights_thread.start()
    resubscription_thread.start()
    
    while not stop_threads:
        time.sleep(1)
    
    stats_consuming_thread.join(1)
    training_thread.join(1)
    pushing_weights_thread.join(1)
    pulling_weights_thread.join(1)
    consumer.close()
    logger.info("Exiting main thread.")
    


if __name__=="__main__":
    main()