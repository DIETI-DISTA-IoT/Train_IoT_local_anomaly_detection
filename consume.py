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
import string
import random
import os


batch_counter = 0
epoch_counter = 0
received_all_real_msg = 0
received_anomalies_msg = 0
received_normal_msg = 0
anomalies_processed = 0
diagnostics_processed = 0
diagnostics_clusters_count = torch.zeros(15)
anomalies_clusters_count = torch.zeros(19)
diagnostics_cluster_percentages =torch.zeros(15)
anomalies_cluster_percentages = torch.zeros(19)
epoch_loss = 0
epoch_accuracy = 0
epoch_precision = 0
epoch_recall = 0
epoch_f1 = 0

def create_consumer():
    def generate_random_string(length=10):
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for i in range(length))
    # Kafka consumer configuration
    conf_cons = {
        'bootstrap.servers': KAFKA_BROKER,  # Kafka broker URL
        'group.id': f'{VEHICLE_NAME}-consumer-group'+generate_random_string(7),  # Consumer group ID for message offset tracking
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
        logger.info(f"Creating missing topics: {[topic.topic for topic in topics_to_create]}")
        result = admin_client.create_topics(topics_to_create)

        for topic, future in result.items():
            try:
                future.result()
                logger.info(f"Topic '{topic}' created successfully.")
            except KafkaException as e:
                logger.error(f"Failed to create topic '{topic}': {e}")


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
        logger.debug(f"received message from topic [{msg.topic()}]")
        return message_value
    except json.JSONDecodeError as e:
        logger.error(f"Error deserializing message: {e}")
        return None


def process_message(topic, msg, producer):
    """
        Process the deserialized message based on its topic.
    """
    global received_all_real_msg, received_anomalies_msg, received_normal_msg, anomalies_processed, diagnostics_processed
    counting_message = False
    # logger.debug(f"Processing message from topic [{topic}]")
    if topic.endswith("_anomalies"):
        anomalies_buffer.add(msg)
        received_anomalies_msg += 1
        anomalies_processed += 1
        counting_message = True
    elif topic.endswith("_normal_data"):
        diagnostics_buffer.add(msg)
        received_normal_msg += 1
        diagnostics_processed += 1
        counting_message = True
    if counting_message:
        received_all_real_msg += 1
    if received_all_real_msg % 500 == 0:
        logger.info(f"Received {received_all_real_msg} messages: {received_anomalies_msg} anomalies, {received_normal_msg} diagnostics.")

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
        weights_copy = brain.get_brain_state_copy()
        weights_reporter.push_weights(weights_copy)


def pull_weights(**kwargs):
    global brain

    while not stop_threads:
        time.sleep(kwargs.get('weights_pull_freq_seconds', 300))        
        new_weights = global_weights_puller.pull_weights()
        if new_weights:
            brain.update_weights(new_weights)
            logger.info("Local weights updated using global model.")


def train_model(**kwargs):
    global brain, diagnostics_processed, anomalies_processed, batch_counter, epoch_counter
    global diagnostics_clusters_count, anomalies_clusters_count, diagnostics_cluster_percentages, anomalies_cluster_percentages
    global epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1

    batch_size = kwargs.get('batch_size', 32)
    epoch_size = kwargs.get('epoch_batches', 50)
    save_model_freq_epochs = kwargs.get('save_model_freq_epochs', 10)

    while not stop_threads:
        batch_feats = None
        batch_labels = None
        batch_preds = None
        do_train_step = False
        batch_loss = 0

        diagnostics_feats, diagnostics_labels, diagnostics_clusters = diagnostics_buffer.sample(batch_size)
        anomalies_feats, anomalies_labels, anomalies_clusters = anomalies_buffer.sample(batch_size)

        if len(diagnostics_feats) > 0:
            batch_feats = diagnostics_feats
            do_train_step = True
            batch_labels = diagnostics_labels

        if len(anomalies_feats) > 0:
            do_train_step = True
            batch_feats = (anomalies_feats if batch_feats is None else torch.vstack((batch_feats, anomalies_feats)))
            batch_labels = (anomalies_labels if batch_labels is None else torch.vstack((batch_labels, anomalies_labels)))

        if do_train_step:
            batch_counter += 1
            batch_preds, loss = brain.train_step(batch_feats, batch_labels)


            # convert bath_preds to binary using pytorch:
            batch_preds = (batch_preds > 0.5).float()
            batch_loss += loss
            batch_accuracy = accuracy_score(batch_labels, batch_preds)
            batch_precision = precision_score(batch_labels, batch_preds, zero_division=0)
            batch_recall = recall_score(batch_labels, batch_preds, zero_division=0)
            batch_f1 = f1_score(batch_labels, batch_preds, zero_division=0)

            if len(diagnostics_clusters) > 0:
                batch_diag_clusters = torch.bincount(diagnostics_clusters.squeeze(-1), minlength=15)
                diagnostics_clusters_count += batch_diag_clusters
                assert diagnostics_processed > 0, "Diagnostics processed is zero."
                diagnostics_cluster_percentages = diagnostics_clusters_count / diagnostics_processed
                        
            if len(anomalies_clusters) > 0:
                batch_anom_clusters = torch.bincount(anomalies_clusters.squeeze(-1), minlength=19)
                anomalies_clusters_count += batch_anom_clusters
                assert anomalies_processed > 0, "Anomalies processed is zero."
                anomalies_cluster_percentages = anomalies_clusters_count / anomalies_processed

            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            epoch_precision += batch_precision
            epoch_recall += batch_recall
            epoch_f1 += batch_f1

            if batch_counter % epoch_size == 0:
                epoch_counter += 1

                epoch_loss /= epoch_size
                epoch_accuracy /= epoch_size
                epoch_precision /= epoch_size
                epoch_recall /= epoch_size
                epoch_f1 /= epoch_size

                metrics_reporter.report({
                    'total_loss': epoch_loss,
                    'accuracy': epoch_accuracy,
                    'precision': epoch_precision,
                    'recall': epoch_recall,
                    'f1': epoch_f1,
                    'diagnostics_processed': diagnostics_processed,
                    'anomalies_processed': anomalies_processed,
                    'diagnostics_cluster_percentages': diagnostics_cluster_percentages.tolist(),
                    'anomalies_cluster_percentages': anomalies_cluster_percentages.tolist()})
                
                epoch_loss = epoch_accuracy = epoch_precision = epoch_recall = epoch_f1 = 0

                if epoch_counter % save_model_freq_epochs == 0:
                    model_path = kwargs.get('model_saving_path', 'default_model.pth')
                    logger.info(f"Saving model after {epoch_counter} epochs as {model_path}.")
                    brain.save_model()

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
    global resubscribe_interval_seconds, epoch_batches

    parser = argparse.ArgumentParser(description='Start the consumer for the specific vehicle.')
    parser.add_argument('--kafka_broker', type=str, default='kafka:9092', help='Kafka broker URL')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Size of the message buffer')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of the batch')
    parser.add_argument('--logging_level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--weights_push_freq_seconds', type=int, default=300, help='Seconds interval between weights push')
    parser.add_argument('--weights_pull_freq_seconds', type=int, default=300, help='Seconds interval between weights pulling from coordinator')
    parser.add_argument('--kafka_topic_update_interval_secs', type=int, default=15, help='Seconds interval between Kafka topic update')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epoch_size', type=int, default=50, help='Number of batches per epoch (for reporting purposes)')
    parser.add_argument('--training_freq_seconds', type=float, default=1, help='Seconds interval between training steps')
    parser.add_argument('--save_model_freq_epochs', type=int, default=10, help='Number of epochs between model saving')
    parser.add_argument('--model_saving_path', type=str, default='default_model.pth', help='Path to save the model')

    args = parser.parse_args()

    VEHICLE_NAME = os.environ.get('VEHICLE_NAME')
    assert VEHICLE_NAME, "VEHICLE_NAME environment variable is not set."
    args.vehicle_name = VEHICLE_NAME

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=str(args.logging_level).upper())
    logger = logging.getLogger(VEHICLE_NAME+'_'+'consumer')

    

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
    
    resubscription_thread.join(1)
    stats_consuming_thread.join(1)
    training_thread.join(1)
    pushing_weights_thread.join(1)
    pulling_weights_thread.join(1)
    consumer.close()
    logger.info("Exiting main thread.")
    


if __name__=="__main__":
    main()