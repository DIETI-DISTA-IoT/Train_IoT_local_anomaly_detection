import argparse
import logging
import threading
import json
import time
from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
import requests
from preprocessing import Buffer, dict_to_tensor
from brain import Brain
from communication import MetricsReporter, WeightsReporter, WeightsPuller
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import signal
import string
import random
import os
import numpy as np
from threading import Lock
from flask import Flask, request, jsonify
import yaml
from OpenFAIR.container_api import ContainerAPI


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

epoch_final_accuracy = 0
epoch_final_precision = 0
epoch_final_recall = 0
epoch_final_f1 = 0

epoch_main_accuracy= 0
epoch_main_precision= 0
epoch_main_recall= 0
epoch_main_f1= 0

epoch_aux_accuracy= 0
epoch_aux_precision= 0
epoch_aux_recall= 0
epoch_aux_f1= 0

average_param = 'binary'

online_final_batch_labels = []
online_final_batch_preds = []
online_main_batch_labels = []
online_main_batch_preds = []
online_aux_batch_labels = []
online_aux_batch_preds = []

mitigation_times = []
mitigation_reward = 0

HOST_IP = os.getenv("HOST_IP")


def thread_safe_lock(lock):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


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
    global received_all_real_msg, received_anomalies_msg, received_normal_msg, anomalies_processed, diagnostics_processed
    counting_message = False
    # logger.debug(f"Processing message from topic [{topic}]")

    if topic.endswith("_anomalies"):
        feat_tensor, final_label_tensor, main_label_tensor, aux_label_tensor, cluster_label_tensor = anomalies_buffer.format(msg)
        anomalies_buffer.add(feat_tensor, final_label_tensor, main_label_tensor, aux_label_tensor, cluster_label_tensor)
        received_anomalies_msg += 1
        anomalies_processed += 1
        counting_message = True
    elif topic.endswith("_normal_data"):
        feat_tensor, final_label_tensor, main_label_tensor, aux_label_tensor, cluster_label_tensor = diagnostics_buffer.format(msg)
        diagnostics_buffer.add(feat_tensor, final_label_tensor, main_label_tensor, aux_label_tensor, cluster_label_tensor)
        received_normal_msg += 1
        diagnostics_processed += 1
        counting_message = True
    if counting_message:
        received_all_real_msg += 1
        online_classification(feat_tensor, final_label_tensor, main_label_tensor, aux_label_tensor)

    if received_all_real_msg % 500 == 0:
        logger.info(f"Received {received_all_real_msg} messages: {received_anomalies_msg} anomalies, {received_normal_msg} diagnostics.")


def mitigation_and_rewarding(prediction, current_label):
    global mitigation_reward
    if prediction == 1:
        if current_label == prediction:
            # True positive.
            if MITIGATION: send_attack_mitigation_request(VEHICLE_NAME)
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


def online_classification(feat_tensor, final_label_tensor, main_label_tensor, aux_label_tensor):
    global online_final_batch_labels, online_final_batch_preds, mitigation_reward
    global online_main_batch_labels, online_main_batch_preds, online_aux_batch_labels, online_aux_batch_preds
    global lists_lock

    brain.model.eval()
    with brain.model_lock, torch.no_grad():
        
        main_pred, aux_pred = brain.model(feat_tensor.unsqueeze(0))

        main_pred = (main_pred > 0.5).float()
        if mode == 'SW':
            aux_pred = (aux_pred > 0.5).float()

            # final_pred = final_pred.argmax(dim=1)

            # approximate final_pred to the closest integer:
            final_pred = torch.round(2*main_pred.detach() + aux_pred.detach())


    # acquire lists lock
    with lists_lock:

        online_main_batch_labels.append(main_label_tensor.float())
        online_main_batch_preds.append(main_pred.squeeze())

        if mode == 'SW':
            online_aux_batch_labels.append(aux_label_tensor.float())
            online_aux_batch_preds.append(aux_pred.squeeze())

            online_final_batch_labels.append(final_label_tensor.float())
            online_final_batch_preds.append(final_pred.squeeze())

    if mode == 'SW': mitigation_and_rewarding(aux_pred, aux_label_tensor)        


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


def send_attack_mitigation_request(vehicle_name):
    global mitigation_times, lists_lock

    url = f"http://{HOST_IP}:{MANAGER_PORT}/stop-attack"
    data = {"vehicle_name": vehicle_name, "origin": "AI"}
    response = requests.post(url, json=data)
    try:
        response_json = response.json()
        logger.debug(f"Mitigate-attack Response JSON: {response_json}")
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
    global epoch_loss, epoch_final_accuracy, epoch_final_precision, epoch_final_recall, epoch_final_f1
    global epoch_main_accuracy, epoch_main_precision, epoch_main_recall, epoch_main_f1
    global epoch_aux_accuracy, epoch_aux_precision, epoch_aux_recall, epoch_aux_f1
    global online_final_batch_labels, online_final_batch_preds, mitigation_reward, mitigation_times
    global online_main_batch_labels, online_main_batch_preds, online_aux_batch_labels, online_aux_batch_preds
    global lists_lock
    
    lists_lock = Lock()

    batch_size = kwargs.get('batch_size', 32)
    epoch_size = kwargs.get('epoch_size', 50)
    save_model_freq_epochs = kwargs.get('save_model_freq_epochs', 10)

    while not stop_threads:
        batch_feats = None
        batch_final_labels = None
        batch_final_preds = None
        batch_main_labels = None
        batch_main_preds = None
        batch_aux_labels = None
        batch_aux_preds = None
        do_train_step = False
        batch_loss = 0

        diagnostics_feats, diag_final_labels, diag_main_labels, diag_aux_labels, diagnostics_clusters = diagnostics_buffer.sample(batch_size)
        anomalies_feats, anom_final_labels, anom_main_labels, anom_aux_labels, anomalies_clusters = anomalies_buffer.sample(batch_size)

        if len(diagnostics_feats) > 0:
            batch_feats = diagnostics_feats
            do_train_step = True
            batch_main_labels = diag_main_labels
            if mode == 'SW':
                batch_final_labels = diag_final_labels
                batch_aux_labels = diag_aux_labels

        if len(anomalies_feats) > 0:
            do_train_step = True
            batch_feats = (anomalies_feats if batch_feats is None else torch.vstack((batch_feats, anomalies_feats)))
            batch_main_labels = (anom_main_labels if batch_main_labels is None else torch.vstack((batch_main_labels, anom_main_labels)))
            if mode == 'SW':
                batch_final_labels = (anom_final_labels if batch_final_labels is None else torch.vstack((batch_final_labels, anom_final_labels)))
                batch_aux_labels = (anom_aux_labels if batch_aux_labels is None else torch.vstack((batch_aux_labels, anom_aux_labels)))

        if do_train_step:
            batch_counter += 1
            batch_final_preds, batch_main_preds, batch_aux_preds, loss = brain.train_step(batch_feats, batch_final_labels, batch_main_labels, batch_aux_labels)

            batch_main_preds = (batch_main_preds > 0.5).float()

            batch_main_accuracy = accuracy_score(batch_main_labels, batch_main_preds)
            batch_main_precision = precision_score(batch_main_labels, batch_main_preds, zero_division=0)
            batch_main_recall = recall_score(batch_main_labels, batch_main_preds, zero_division=0)
            batch_main_f1 = f1_score(batch_main_labels, batch_main_preds, zero_division=0)

            if mode == 'SW':
                batch_aux_preds = (batch_aux_preds > 0.5).float()
                

                batch_aux_accuracy = accuracy_score(batch_aux_labels, batch_aux_preds)
                batch_aux_precision = precision_score(batch_aux_labels, batch_aux_preds, zero_division=0)
                batch_aux_recall = recall_score(batch_aux_labels, batch_aux_preds, zero_division=0)
                batch_aux_f1 = f1_score(batch_aux_labels, batch_aux_preds, zero_division=0)

                # batch_final_preds = torch.argmax(batch_final_preds, dim=1)

                batch_final_accuracy = accuracy_score(batch_final_labels, batch_final_preds)
                batch_final_precision = precision_score(batch_final_labels, batch_final_preds, zero_division=0, average=average_param)
                batch_final_recall = recall_score(batch_final_labels, batch_final_preds, zero_division=0, average=average_param)
                batch_final_f1 = f1_score(batch_final_labels, batch_final_preds, zero_division=0, average=average_param)


            batch_loss += loss
            
            if len(diagnostics_clusters) > 0:
                labels = diagnostics_clusters.squeeze(-1)
                labels = labels[labels >= 0].to(torch.long)
                if labels.numel() > 0:
                    batch_diag_clusters = torch.bincount(labels, minlength=15)
                    diagnostics_clusters_count += batch_diag_clusters
                    diagnostics_cluster_percentages = diagnostics_clusters_count / diagnostics_clusters_count.sum()
                        
            if len(anomalies_clusters) > 0:
                labels = anomalies_clusters.squeeze(-1)
                labels = labels[labels >= 0].to(torch.long)
                if labels.numel() > 0:
                    batch_anom_clusters = torch.bincount(labels, minlength=19)
                    anomalies_clusters_count += batch_anom_clusters
                    anomalies_cluster_percentages = anomalies_clusters_count / anomalies_clusters_count.sum()

            epoch_loss += batch_loss

            epoch_main_accuracy += batch_main_accuracy
            epoch_main_precision += batch_main_precision
            epoch_main_recall += batch_main_recall
            epoch_main_f1 += batch_main_f1

            if mode == 'SW':
                epoch_aux_accuracy += batch_aux_accuracy
                epoch_aux_precision += batch_aux_precision
                epoch_aux_recall += batch_aux_recall
                epoch_aux_f1 += batch_aux_f1

                epoch_final_accuracy += batch_final_accuracy
                epoch_final_precision += batch_final_precision
                epoch_final_recall += batch_final_recall
                epoch_final_f1 += batch_final_f1

            if batch_counter % epoch_size == 0:
                
                epoch_counter += 1

                epoch_loss /= epoch_size

                epoch_main_accuracy /= epoch_size
                epoch_main_precision /= epoch_size
                epoch_main_recall /= epoch_size
                epoch_main_f1 /= epoch_size

                if mode == 'SW':
                    epoch_aux_accuracy /= epoch_size
                    epoch_aux_precision /= epoch_size
                    epoch_aux_recall /= epoch_size
                    epoch_aux_f1 /= epoch_size

                    epoch_final_accuracy /= epoch_size
                    epoch_final_precision /= epoch_size
                    epoch_final_recall /= epoch_size
                    epoch_final_f1 /= epoch_size

                metrics_dict = {
                    'total_loss': epoch_loss,
                    'class_accuracy': epoch_main_accuracy,
                    'class_precision': epoch_main_precision,
                    'class_recall': epoch_main_recall,
                    'class_f1': epoch_main_f1,
                    'diagnostics_processed': diagnostics_processed,
                    'anomalies_processed': anomalies_processed,
                    'diagnostics_cluster_percentages': diagnostics_cluster_percentages.tolist(),
                    'anomalies_cluster_percentages': anomalies_cluster_percentages.tolist()
                }

                if mode == 'SW':
                    metrics_dict['attack_accuracy'] = epoch_aux_accuracy
                    metrics_dict['attack_precision'] = epoch_aux_precision
                    metrics_dict['attack_recall'] = epoch_aux_recall
                    metrics_dict['attack_f1'] = epoch_aux_f1

                    metrics_dict['accuracy'] = epoch_final_accuracy
                    metrics_dict['precision'] = epoch_final_precision
                    metrics_dict['recall'] = epoch_final_recall
                    metrics_dict['f1'] = epoch_final_f1
                
                if len(online_main_batch_labels) > 20:
                    with lists_lock:
                        online_main_batch_accuracy = accuracy_score(online_main_batch_labels, online_main_batch_preds)
                        online_main_batch_precision = precision_score(online_main_batch_labels, online_main_batch_preds, zero_division=0)
                        online_main_batch_recall = recall_score(online_main_batch_labels, online_main_batch_preds, zero_division=0)
                        online_main_batch_f1 = f1_score(online_main_batch_labels, online_main_batch_preds, zero_division=0)

                        if mode == 'SW':
                            online_final_batch_accuracy = accuracy_score(online_final_batch_labels, online_final_batch_preds)
                            online_final_batch_precision = precision_score(online_final_batch_labels, online_final_batch_preds, zero_division=0, average=average_param)
                            online_final_batch_recall = recall_score(online_final_batch_labels, online_final_batch_preds, zero_division=0, average=average_param)
                            online_final_batch_f1 = f1_score(online_final_batch_labels, online_final_batch_preds, zero_division=0, average=average_param)


                            online_aux_batch_accuracy = accuracy_score(online_aux_batch_labels, online_aux_batch_preds)
                            online_aux_batch_precision = precision_score(online_aux_batch_labels, online_aux_batch_preds, zero_division=0)
                            online_aux_batch_recall = recall_score(online_aux_batch_labels, online_aux_batch_preds, zero_division=0)
                            online_aux_batch_f1 = f1_score(online_aux_batch_labels, online_aux_batch_preds, zero_division=0)

                        online_metrics_dict = {
                            'online_class_accuracy': online_main_batch_accuracy,
                            'online_class_precision': online_main_batch_precision,
                            'online_class_recall': online_main_batch_recall,
                            'online_class_f1': online_main_batch_f1
                            }
                        
                        if mode == 'SW':
                            online_metrics_dict.update({
                                'online_accuracy': online_final_batch_accuracy,
                                'online_precision': online_final_batch_precision,
                                'online_recall': online_final_batch_recall,
                                'online_f1': online_final_batch_f1,
                                'online_attack_accuracy': online_aux_batch_accuracy,
                                'online_attack_precision': online_aux_batch_precision,
                                'online_attack_recall': online_aux_batch_recall,
                                'online_attack_f1': online_aux_batch_f1
                                })

                            online_metrics_dict['mitigation_time'] = np.array(mitigation_times).mean() if len(mitigation_times) > 0 else 0.0
                            online_metrics_dict['mitigation_reward'] = mitigation_reward

                        metrics_dict.update(online_metrics_dict)

                        online_final_batch_labels = []
                        online_final_batch_preds = []
                        online_main_batch_labels = []
                        online_main_batch_preds = []
                        online_aux_batch_labels = []
                        online_aux_batch_preds = []
                        mitigation_times = []
                        mitigation_reward = 0


                metrics_reporter.report(metrics_dict)
                
                epoch_loss = epoch_final_accuracy = epoch_final_precision = epoch_final_recall = epoch_final_f1 = 0
                epoch_main_accuracy = epoch_main_precision = epoch_main_recall = epoch_main_f1 = 0
                epoch_aux_accuracy = epoch_aux_precision = epoch_aux_recall = epoch_aux_f1 = 0

                diagnostics_clusters_count = torch.zeros(15)
                anomalies_clusters_count = torch.zeros(19)
                    

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


def parse_str_list(arg):
    # Split the input string by commas and convert each element to int
    try:
        return [str(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Arguments must be strings separated by commas")
    

def configure_no_proxy():
    os.environ['no_proxy'] = os.environ.get('no_proxy', '') + f",{HOST_IP}"


def build_args_from_config(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kafka_broker', type=str, default='kafka:9092')
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--weights_push_freq_seconds', type=int, default=300)
    parser.add_argument('--weights_pull_freq_seconds', type=int, default=300)
    parser.add_argument('--kafka_topic_update_interval_secs', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epoch_size', type=int, default=50)
    parser.add_argument('--training_freq_seconds', type=float, default=1)
    parser.add_argument('--save_model_freq_epochs', type=int, default=10)
    parser.add_argument('--model_saving_path', type=str, default='default_model.pth')
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--input_dim', type=int, default=59)
    parser.add_argument('--mode', type=str, default='OF')
    parser.add_argument('--probe_metrics', type=parse_str_list, default=['RTT','INBOUND','OUTBOUND','CPU','MEM'])
    parser.add_argument('--mitigation', action='store_true')
    parser.add_argument('--true_positive_reward', type=float, default=2.0)
    parser.add_argument('--true_negative_reward', type=float, default=0)
    parser.add_argument('--false_positive_reward', type=float, default=-4)
    parser.add_argument('--false_negative_reward', type=float, default=-10)
    parser.add_argument('--no_proxy_host', action='store_true')
    parser.add_argument('--manager_port', type=int, default=5000)

    # Convert config dict to args list
    args_list = []
    for k, v in config.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                args_list.append(flag)
        elif isinstance(v, list):
            if k == 'probe_metrics':
                args_list.extend([flag, ",".join(map(str, v))])
            else:
                continue
        else:
            args_list.extend([flag, str(v)])
    return parser.parse_args(args_list)


def start_consumer_runtime(args_namespace):
    global VEHICLE_NAME, KAFKA_BROKER, MANAGER_PORT, MITIGATION, mode, average_param
    global batch_size, stop_threads, stats_consuming_thread, training_thread, pushing_weights_thread, pulling_weights_thread
    global anomalies_buffer, diagnostics_buffer, brain, metrics_reporter, logger, weights_reporter, global_weights_puller
    global resubscribe_interval_seconds, epoch_batches
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

    mode = args.mode
    if mode == 'SW':
        args.output_dim = 4
        average_param = 'macro'

    VEHICLE_NAME = os.environ.get('VEHICLE_NAME')
    assert VEHICLE_NAME, "VEHICLE_NAME environment variable is not set."
    args.vehicle_name = VEHICLE_NAME

    logging.basicConfig(format='%(name)s-%(levelname)s-%(message)s', level=str(args.logging_level).upper())
    logger = logging.getLogger(f'[{VEHICLE_NAME}_CONS]')

    KAFKA_BROKER = args.kafka_broker

    print(f"Starting consumer for vehicle {VEHICLE_NAME}")

    brain = Brain(**vars(args))
    metrics_reporter = MetricsReporter(**vars(args))
    weights_reporter = WeightsReporter(**vars(args))
    global_weights_puller = WeightsPuller(**vars(args))

    anomalies_buffer = Buffer(args.buffer_size, label=1, mode=mode)
    diagnostics_buffer = Buffer(args.buffer_size, label=0, mode=mode)

    resubscribe_interval_seconds = args.kafka_topic_update_interval_secs
    resubscription_thread = threading.Thread(target=resubscribe)
    resubscription_thread.daemon = True

    stats_consuming_thread = threading.Thread(target=consume_vehicle_data)
    stats_consuming_thread.daemon = True

    training_thread = threading.Thread(target=train_model, kwargs=vars(args))
    training_thread.daemon = True

    pushing_weights_thread = threading.Thread(target=push_weights, kwargs=vars(args))
    pushing_weights_thread.daemon = True

    pulling_weights_thread = threading.Thread(target=pull_weights, kwargs=vars(args))
    pulling_weights_thread.daemon = True

    # Avoid setting signal handlers from within Flask request thread
    stop_threads = False

    stats_consuming_thread.start()
    training_thread.start()
    pushing_weights_thread.start()
    pulling_weights_thread.start()
    resubscription_thread.start()

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
    try:
        threads_dict['resubscription_thread'].join(1)
        threads_dict['stats_consuming_thread'].join(1)
        threads_dict['training_thread'].join(1)
        threads_dict['pushing_weights_thread'].join(1)
        threads_dict['pulling_weights_thread'].join(1)
    except Exception:
        pass
    try:
        consumer.close()
    except Exception:
        pass
    try:
        logger.info("Exiting main thread.")
    except Exception:
        pass


class ConsumerAPI(ContainerAPI):
    def __init__(self, container_name: str, port: int = 5000):
        super().__init__(container_type='consumer', container_name=container_name, port=port)
        self._threads = None

    def validate_config(self, config):
        if 'kafka_broker' not in config:
            config['kafka_broker'] = 'kafka:9092'
        return True

    def handle_start(self, data):
        if self._threads is not None:
            return {'status': 'already_running'}
        args = build_args_from_config(self.config)
        runtime = start_consumer_runtime(args)
        self._threads = runtime['threads']
        return {'status': 'started', 'vehicle': os.getenv('VEHICLE_NAME')}

    def handle_stop(self, data):
        if self._threads is None:
            return {'status': 'already_stopped'}
        shutdown_runtime(self._threads)
        self._threads = None
        return {'status': 'stopped'}


def main():
    api = ConsumerAPI(container_name=os.getenv('VEHICLE_NAME') or 'unknown_consumer', port=5000)
    api.run()
    


if __name__=="__main__":
    main()