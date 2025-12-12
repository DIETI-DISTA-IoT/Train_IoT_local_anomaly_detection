import argparse
import logging
import threading
import json
import time
from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
import requests
from preprocessing import Buffer
from brain import Brain
from communication import MetricsReporter, WeightsReporter, WeightsPuller
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import string
import random
import os
import numpy as np
from threading import Lock
from flask import Flask
from OpenFAIR.container_api import ContainerAPI
from OpenFAIR import EventType
import wandb
import matplotlib.pyplot as plt

batch_counter = 0
epoch_counter = 0
records_processed = 0
attacks_processed = 0
anomalies_processed = 0
diagnostics_processed = 0
eval_anomalies_processed = 0
eval_attacks_processed = 0

epoch_loss = 0

epoch_accuracy= 0
epoch_precision= 0
epoch_recall= 0
epoch_f1= 0

average_param = 'binary'

online_batch_labels = []
online_main_batch_preds = []

mitigation_times = []
mitigation_reward = 0

HOST_IP = os.getenv("HOST_IP")

columns_to_delete = ['Flotta', 'Veicolo', 'Codice', 'Nome', 'Descrizione', 'Timestamp', 'Timestamp chiusura', 'Durata', 
                        'Posizione', 'Sistema', 'Componente', 'Timestamp segnale', 'Test']


def encode_array(arr):
    return {
        "data": arr.tobytes().hex(),
        "shape": arr.shape,
        "dtype": str(arr.dtype)
    }


def thread_safe_lock(lock):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def visual_evaluation(n=1000):
    global brain
    diagnostics_feats, diag_main_labels = diagnostics_buffer.sample(n // 3)
    anomalies_feats, anom_main_labels = eval_anomalies_buffer.sample(n // 3)
    attack_feats, attack_main_labels = eval_attacks_buffer.sample(n // 3)
    feats = torch.vstack((diagnostics_feats, anomalies_feats, attack_feats))
    y = torch.vstack((diag_main_labels, anom_main_labels, attack_main_labels))
    brain.model.eval()
    with brain.model_lock, torch.no_grad():
        preds, manifold = brain.model(feats)
        preds = preds.argmax(dim=1)


    # PCA using torch only (2 components)
    X = feats - feats.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(X, q=2)
    X2 = X @ V[:, :2]

    # Return the projected data so the caller can plot it externally
    # return plot_results(y, preds, X2, manifold, VEHICLE_NAME)
    return {
    'visual_eval_X': encode_array(X2.numpy()),
    'visual_eval_y': encode_array(y.numpy()),
    'visual_eval_preds': encode_array(preds.numpy()),
    'visual_eval_manifold': encode_array(manifold.numpy())
    }


def plot_results(Y, all_preds, pca_embed, manifold, task_name):

        _, axes = plt.subplots(1, 3, figsize=(20, 4))

        colors = ['r', 'g', 'b']

        # First subplot
        ax = axes[0]
        for eventype in EventType:
            mask = Y.squeeze() == eventype.value
            ax.scatter(pca_embed[mask, 0], pca_embed[mask, 1],
                    c=colors[eventype.value], s=15, alpha=0.1, label=eventype.name)
        ax.set_title(f'Input-Space (2D-PCA) {task_name}')
        ax.legend()

        # Second subplot
        ax = axes[1]
        for eventype in EventType:
            mask = Y.squeeze() == eventype.value
            ax.scatter(manifold[mask, 0], manifold[mask, 1],
                    c=colors[eventype.value], s=15, alpha=0.1, label=eventype.name)
        ax.set_title(f'2D-Representation-Space (labels) {task_name}')
        ax.legend()
        

        # Third subplot
        ax = axes[2]
        for eventype in EventType:
            mask = all_preds == eventype.value
            ax.scatter(manifold[mask, 0], manifold[mask, 1],
                    c=colors[eventype.value], s=15, alpha=0.1, label=eventype.name)
        ax.set_title(f'Predictions {task_name}')
        ax.legend()
        plt.savefig(f'{task_name}_manifold_projection.png')
        return plt


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


def process_message(topic, msg):
    """
        Process the deserialized message based on its topic.
    """
    global records_processed
    global anomalies_processed, diagnostics_processed, attacks_processed
    global anomalies_buffer, diagnostics_buffer, attacks_buffer
    global eval_anomalies_buffer, eval_attacks_buffer
    global eval_anomalies_processed, eval_attacks_processed


    counting_message = False
    # logger.debug(f"Processing message from topic [{topic}]")

    for col in columns_to_delete:
        if col in msg:
            del msg[col]

    if topic.endswith("_eval_anomalies"):
        if msg['event_type'] == EventType.ANOMALY.value:
            feat_tensor, main_label_tensor = eval_anomalies_buffer.format(msg)
            eval_anomalies_buffer.add(feat_tensor, main_label_tensor)
            eval_anomalies_processed += 1
        elif msg['event_type'] == EventType.ATTACK.value:
            feat_tensor, main_label_tensor = eval_attacks_buffer.format(msg)
            eval_attacks_buffer.add(feat_tensor, main_label_tensor)
            eval_attacks_processed += 1

    elif topic.endswith("_anomalies"):
        counting_message = True
        if msg['event_type'] == EventType.ANOMALY.value:
            feat_tensor, main_label_tensor = anomalies_buffer.format(msg)
            anomalies_buffer.add(feat_tensor, main_label_tensor)
            anomalies_processed += 1

        elif msg['event_type'] == EventType.ATTACK.value:
            feat_tensor, main_label_tensor = attacks_buffer.format(msg)
            attacks_buffer.add(feat_tensor, main_label_tensor)
            attacks_processed += 1

    elif topic.endswith("_normal_data"):
        counting_message = True
        feat_tensor, main_label_tensor = diagnostics_buffer.format(msg)
        diagnostics_buffer.add(feat_tensor, main_label_tensor)
        diagnostics_processed += 1
        
    if counting_message:
        records_processed += 1
        online_classification(feat_tensor, main_label_tensor)

    if records_processed % 500 == 0:
        logger.info(f"Received {records_processed} messages: {attacks_processed} attacks, {anomalies_processed} anomalies, {diagnostics_processed} diagnostics.")
        logger.info(f"Received {eval_anomalies_processed} eval_anomalies, {eval_attacks_processed} eval_attacks.")

def send_attack_mitigation_request(vehicle_name):
    global mitigation_times, lists_lock

    url = f"http://{HOST_IP}:{MANAGER_PORT}/stop-attack"
    data = {"vehicle_name": vehicle_name, "origin": "AI"}
    response = requests.post(url, json=data)
    try:
        response_json = response.json()
        logger.info(f"Mitigate-attack Response JSON: {response_json}")
        mitigation_time = response_json.get('mitigation_time')
        if mitigation_time is not None:
            with lists_lock:
                mitigation_times.append(mitigation_time)
        else:
            msg = response_json.get('message')
            assert msg is not None
            logger.warning(f"Mitigation req. failed. Answer: {msg}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from response: {e}")
        response_json = {}


def get_status_from_manager(vehicle_name):
    url = f"http://{HOST_IP}:{MANAGER_PORT}/vehicle-status"
    data = {"vehicle_name": vehicle_name}
    response = requests.post(url, json=data)
    logger.debug(f"Vehicle-status Response Status Code: {response.status_code}")
    logger.debug(f"Vehicle-status Response Body: {response.text}")
    return response.text


def mitigation_and_rewarding(prediction, current_label):
    global mitigation_reward
    if prediction == 2:
        if current_label == prediction:
            # True positive.
            if MITIGATION:
                if get_status_from_manager(VEHICLE_NAME) == "INFECTED":
                    send_attack_mitigation_request(VEHICLE_NAME)
            mitigation_reward += true_positive_reward
        else:
            # False positive
            mitigation_reward += false_positive_reward
    else:
        if current_label == prediction:
            # True negative
            mitigation_reward += true_negative_reward
        else:
            # False negative
            mitigation_reward += false_negative_reward


def online_classification(feat_tensor, main_label_tensor):
    global online_final_batch_preds, mitigation_reward
    global online_batch_labels, online_main_batch_preds
    global lists_lock

    brain.model.eval()
    with brain.model_lock, torch.no_grad():
        main_pred, _ = brain.model(feat_tensor.unsqueeze(0))
        main_pred = main_pred.argmax(dim=1)
        
    with lists_lock:
        online_batch_labels.append(main_label_tensor)
        online_main_batch_preds.append(main_pred.squeeze())

    mitigation_and_rewarding(main_pred, main_label_tensor)        


def subscribe_to_topics():
    """
        Subscribe to a list of Kafka topics.
    """
    global consumer

    topics = [f"{VEHICLE_NAME}_anomalies", f"{VEHICLE_NAME}_eval_anomalies" ,f"{VEHICLE_NAME}_normal_data"]
    consumer.subscribe(topics)
    global_weights_puller.subscribe()
    logger.debug(f"(re)subscribed to topics: {topics}")


def consume_vehicle_data():
    """
        Consume messages for a specific vehicle from Kafka topics.
    """
    global consumer

    stats_topic= f"{VEHICLE_NAME}_statistics"
    weights_topic = f"{VEHICLE_NAME}_weights"

    check_and_create_topics([stats_topic, weights_topic])

    consumer = create_consumer()

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
                process_message(msg.topic(), deserialized_data)

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
    global brain, batch_counter, epoch_counter
    global epoch_loss
    global epoch_accuracy, epoch_precision, epoch_recall, epoch_f1
    global mitigation_reward, mitigation_times
    global online_batch_labels, online_main_batch_preds
    global lists_lock
    
    lists_lock = Lock()

    batch_size = kwargs.get('batch_size', 32)
    epoch_size = kwargs.get('epoch_size', 50)
    save_model_freq_epochs = kwargs.get('save_model_freq_epochs', 10)

    while not stop_threads:
        batch_feats = None
        batch_main_labels = None
        batch_main_preds = None
        batch_loss = 0

        diagnostics_feats, diag_main_labels = diagnostics_buffer.sample(batch_size)
        anomalies_feats, anom_main_labels = anomalies_buffer.sample(batch_size)
        attack_feats, attack_main_labels = attacks_buffer.sample(batch_size)

        if adversarial_training:
            adv_anomalies_feats, adv_anom_main_labels = eval_anomalies_buffer.sample(batch_size)
            adv_attack_feats, adv_attack_main_labels = eval_attacks_buffer.sample(batch_size)

        if len(diagnostics_feats) >= batch_size and len(anomalies_feats) >= batch_size and len(attack_feats) >= batch_size:

            if adversarial_training:
                batch_feats = torch.vstack((diagnostics_feats, anomalies_feats, attack_feats, adv_anomalies_feats, adv_attack_feats))
                batch_main_labels = torch.vstack((diag_main_labels, anom_main_labels, attack_main_labels, adv_anom_main_labels, adv_attack_main_labels))
            else:
                batch_feats = torch.vstack((diagnostics_feats, anomalies_feats, attack_feats))
                batch_main_labels = torch.vstack((diag_main_labels, anom_main_labels, attack_main_labels))
        
            batch_counter += 1
            batch_logits, batch_loss = brain.train_step(batch_feats, batch_main_labels)
            batch_main_preds = batch_logits.argmax(dim=1)

            batch_accuracy = accuracy_score(batch_main_labels, batch_main_preds)
            batch_precision = precision_score(batch_main_labels, batch_main_preds, zero_division=0, average='weighted')
            batch_recall = recall_score(batch_main_labels, batch_main_preds, zero_division=0, average='weighted')
            batch_f1 = f1_score(batch_main_labels, batch_main_preds, zero_division=0, average='weighted')            
            
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

                metrics_dict = {
                    'total_loss': epoch_loss,
                    'class_accuracy': epoch_accuracy,
                    'class_precision': epoch_precision,
                    'class_recall': epoch_recall,
                    'class_f1': epoch_f1,
                    'diagnostics_processed': diagnostics_processed,
                    'anomalies_processed': anomalies_processed,
                    'attacks_processed': attacks_processed,
                    'records_processed': records_processed
                }
                
                if len(online_batch_labels) > 20:
                    with lists_lock:
                        online_main_batch_accuracy = accuracy_score(online_batch_labels, online_main_batch_preds)
                        online_main_batch_precision = precision_score(online_batch_labels, online_main_batch_preds, zero_division=0, average='weighted')
                        online_main_batch_recall = recall_score(online_batch_labels, online_main_batch_preds, zero_division=0, average='weighted')
                        online_main_batch_f1 = f1_score(online_batch_labels, online_main_batch_preds, zero_division=0, average='weighted')

                        online_metrics_dict = {
                            'online_class_accuracy': online_main_batch_accuracy,
                            'online_class_precision': online_main_batch_precision,
                            'online_class_recall': online_main_batch_recall,
                            'online_class_f1': online_main_batch_f1
                            }
                        
                        online_metrics_dict['mitigation_time'] = np.array(mitigation_times).mean() if len(mitigation_times) > 0 else 0.0
                        online_metrics_dict['mitigation_reward'] = mitigation_reward

                        metrics_dict.update(online_metrics_dict)

                        online_batch_labels = []
                        online_main_batch_preds = []
                        mitigation_times = []
                        mitigation_reward = 0


                metrics_reporter.report(metrics_dict)
                
                epoch_loss = epoch_accuracy = epoch_precision = epoch_recall = epoch_f1 = 0


                if epoch_counter % save_model_freq_epochs == 0:
                    model_path = kwargs.get('model_saving_path', 'default_model.pth')
                    logger.info(f"Saving model after {epoch_counter} epochs as {model_path}.")
                    brain.save_model()
                    visual_eval_dict = visual_evaluation()
                    logger.info(f"Sending visual evaluation results to wandber...")
                    metrics_reporter.report(visual_eval_dict)

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


def parse_str_list(arg):
    # Split the input string by commas and convert each element to int
    try:
        return [str(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Arguments must be strings separated by commas")
    

def configure_no_proxy():
    os.environ['no_proxy'] = os.environ.get('no_proxy', '') + f",{HOST_IP}"


def start_consumer_runtime(args_namespace):
    global VEHICLE_NAME, KAFKA_BROKER, MANAGER_PORT, MITIGATION, mode, average_param
    global batch_size, stop_threads, stats_consuming_thread, training_thread, pushing_weights_thread, pulling_weights_thread
    global attacks_buffer, anomalies_buffer, diagnostics_buffer, brain, metrics_reporter, logger, weights_reporter, global_weights_puller
    global eval_attacks_buffer, eval_anomalies_buffer, adversarial_training
    global resubscribe_interval_seconds, epoch_batches, adversarial_degree
    global true_positive_reward, false_positive_reward, true_negative_reward, false_negative_reward

    args = args_namespace

    MITIGATION = args.mitigation
    MANAGER_PORT = args.manager_port

    true_positive_reward = args.true_positive_reward
    true_negative_reward = args.true_negative_reward
    false_positive_reward = args.false_positive_reward
    false_negative_reward = args.false_negative_reward

    if args.no_proxy_host:
        configure_no_proxy()
    
    args.output_dim = 3

    VEHICLE_NAME = os.environ.get('VEHICLE_NAME')
    assert VEHICLE_NAME, "VEHICLE_NAME environment variable is not set."
    args.vehicle_name = VEHICLE_NAME

    logging.basicConfig(format='%(name)s-%(levelname)s-%(message)s', level=str(args.logging_level).upper())
    logger = logging.getLogger(f'[{VEHICLE_NAME}_CONS]')

    KAFKA_BROKER = args.kafka_broker

    logger.info(f"Starting consumer for vehicle {VEHICLE_NAME} with adversarial evaluation degree {args.adversarial_degree}")
    logger.info(f"Adversarial training: {args.adversarial_training}")
    adversarial_degree = args.adversarial_degree
    adversarial_training = args.adversarial_training

    logger.info(f"Starting consumer for vehicle {VEHICLE_NAME}")
    logger.info(f"Starting brain for vehicle {VEHICLE_NAME}")
    brain = Brain(**vars(args))
    logger.info(f"Starting metrics reporter for vehicle {VEHICLE_NAME}")
    metrics_reporter = MetricsReporter(**vars(args))
    logger.info(f"Starting weights reporter for vehicle {VEHICLE_NAME}")
    weights_reporter = WeightsReporter(**vars(args))
    logger.info(f"Starting global weights puller for vehicle {VEHICLE_NAME}")
    global_weights_puller = WeightsPuller(**vars(args))

    attacks_buffer = Buffer(args.buffer_size)
    eval_attacks_buffer = Buffer(args.buffer_size)
    anomalies_buffer = Buffer(args.buffer_size)
    eval_anomalies_buffer = Buffer(args.buffer_size)
    diagnostics_buffer = Buffer(args.buffer_size)

    resubscribe_interval_seconds = args.kafka_topic_update_interval_secs
    resubscription_thread = threading.Thread(target=resubscribe)
    resubscription_thread.daemon = True

    stats_consuming_thread = threading.Thread(target=consume_vehicle_data)
    stats_consuming_thread.daemon = True
    logger.info(f"Starting stats consuming thread for vehicle {VEHICLE_NAME}")

    training_thread = threading.Thread(target=train_model, kwargs=vars(args))
    training_thread.daemon = True
    logger.info(f"Starting training thread for vehicle {VEHICLE_NAME}")

    pushing_weights_thread = threading.Thread(target=push_weights, kwargs=vars(args))
    pushing_weights_thread.daemon = True
    logger.info(f"Starting pushing weights thread for vehicle {VEHICLE_NAME}")

    pulling_weights_thread = threading.Thread(target=pull_weights, kwargs=vars(args))
    pulling_weights_thread.daemon = True
    logger.info(f"Starting pulling weights thread for vehicle {VEHICLE_NAME}")

    # Avoid setting signal handlers from within Flask request thread
    stop_threads = False

    stats_consuming_thread.start()
    training_thread.start()
    pushing_weights_thread.start()
    pulling_weights_thread.start()
    resubscription_thread.start()
    logger.info(f"Consumer runtime started for vehicle {VEHICLE_NAME}")

    return {
        'threads': {
            'resubscription_thread': resubscription_thread,
            'stats_consuming_thread': stats_consuming_thread,
            'training_thread': training_thread,
            'pushing_weights_thread': pushing_weights_thread,
            'pulling_weights_thread': pulling_weights_thread
        }
    }


def shutdown_runtime(threads_dict):
    global stop_threads, consumer, logger
    stop_threads = True
    logger.info("Stopping consumer runtime...")
    try:
        logger.info("Waiting for threads to stop...")
        threads_dict['resubscription_thread'].join(1)
        threads_dict['stats_consuming_thread'].join(1)
        threads_dict['training_thread'].join(1)
        threads_dict['pushing_weights_thread'].join(1)
        threads_dict['pulling_weights_thread'].join(1)
        logger.info("Threads stopped.")
    except Exception as e:
        logger.error(f"Error stopping threads: {e}")
        pass
    try:
        logger.info("Closing Kafka consumer...")
        consumer.close()
    except Exception as e:
        logger.error(f"Error closing Kafka consumer: {e}")
        pass
    logger.info("Exiting main thread.")
    

class ConsumerAPI(ContainerAPI):
    def __init__(self, container_name: str, port: int = 5000):
        super().__init__(container_type='consumer', container_name=container_name, port=port)
        self._threads = None
        self.logger.info("ConsumerAPI initialized.")


    def validate_config(self, config):
        if 'kafka_broker' not in config:
            config['kafka_broker'] = 'kafka:9092'
        return True

    def handle_start(self, data):
        self.logger.info("Starting consumer...")
        if self._threads is not None:
            self.logger.info("Consumer is already running.")
            return {'status': 'already_running'}
        runtime = start_consumer_runtime(argparse.Namespace(**self.config))
        self._threads = runtime['threads']
        self.logger.info("Consumer started.")
        return {'status': 'started', 'vehicle': os.getenv('VEHICLE_NAME')}

    def handle_stop(self, data):
        self.logger.info("Stopping consumer...")
        if self._threads is None:
            self.logger.info("Consumer is already stopped.")
            return {'status': 'already_stopped'}
        shutdown_runtime(self._threads)
        self._threads = None
        self.logger.info("Consumer stopped.")
        return {'status': 'stopped'}


def main():
    api = ConsumerAPI(container_name=os.getenv('VEHICLE_NAME') or 'unknown_consumer', port=5000)
    api.run()
    


if __name__=="__main__":
    main()