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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
torch.backends.mkldnn.enabled = False
torch.backends.nnpack.enabled = False
import string
import random
import os
import numpy as np
from threading import Lock
from OpenFAIR.container_api import ContainerAPI
from OpenFAIR import EventType
import matplotlib.pyplot as plt
from hopskipjump import hopskipjump_attack

# Indices, within the 40-dim feature vector produced by
# OpenFAIR.train_simulator.Train.step(), of usBpPres and usMpPres (the brake
# pipe and main reservoir pressures, components 1-2). Opt-in subset (via the
# 'hsja_feature_indices' config kwarg) matching the threat model of the
# Gaussian-noise adversarial stress test (Mp_std/Bp_std in
# train_simulator.generate_hydraulics, which only perturb these two scalars).
# Default HSJA behaviour (feature_indices=None) perturbs the whole vector.
HSJA_PRESSURE_FEATURE_INDICES = [32, 33]

batch_counter = 0
epoch_counter = 0
records_processed = 0
attacks_processed = 0
anoms_processed = 0
diagnostics_processed = 0
eval_anomalies_processed = 0
eval_attacks_processed = 0

epoch_loss = 0

epoch_accuracy= 0
epoch_precision= 0
epoch_recall= 0
epoch_f1= 0
epoch_macro_f1= 0

average_param = 'binary'

online_batch_labels = []
online_main_batch_preds = []

mitigation_times = []
mitigation_reward = 0

HOST_IP = os.getenv("HOST_IP")
MANAGER_IP = None

# Background HSJA evaluation state
_hsja_eval_running = False
_hsja_eval_thread = None

# Counts how many times the adversarial benchmarks (Gaussian-noise eval + HSJA)
# have run, used to decide when to additionally send plots/confusion matrices.
benchmark_eval_counter = 0

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


def visual_evaluation(n=1000, include_plots=True):
    global brain
    diagnostics_feats, diag_main_labels = diagnostics_buffer.sample(n // 3)
    anomalies_feats, anom_main_labels = eval_anomalies_buffer.sample(n // 3)
    attack_feats, attack_main_labels = eval_attacks_buffer.sample(n // 3)

    if len(diagnostics_feats) < 10 or len(anomalies_feats) < 10 or len(attack_feats) < 10:
        return None

    feats = torch.vstack((diagnostics_feats, anomalies_feats, attack_feats))
    y = torch.vstack((diag_main_labels, anom_main_labels, attack_main_labels))
    brain.model.eval()
    with brain.model_lock, torch.no_grad():
        preds, manifold = brain.model(feats)
        preds = preds.argmax(dim=1)

    y = y.numpy()
    preds = preds.numpy()

    adv_eval_accuracy = accuracy_score(y, preds)
    adv_eval_precision = precision_score(y, preds, zero_division=0, average='weighted')
    adv_eval_recall = recall_score(y, preds, zero_division=0, average='weighted')
    adv_eval_f1 = f1_score(y, preds, zero_division=0, average='weighted')
    adv_eval_macro_f1 = f1_score(y, preds, zero_division=0, average='macro')

    result = {
        'adv_eval_accuracy': adv_eval_accuracy,
        'adv_eval_precision': adv_eval_precision,
        'adv_eval_recall': adv_eval_recall,
        'adv_eval_f1': adv_eval_f1,
        'adv_eval_macro_f1': adv_eval_macro_f1,
    }

    if include_plots:
        adv_eval_cm = confusion_matrix(y, preds, labels=[0, 1, 2])

        X = feats - feats.mean(0, keepdim=True)
        U, S, V = torch.pca_lowrank(X, q=2)
        X2 = X @ V[:, :2]

        result.update({
            'visual_eval_X': encode_array(X2.numpy()),
            'visual_eval_y': encode_array(y),
            'visual_eval_preds': encode_array(preds),
            'visual_eval_manifold': encode_array(manifold.numpy()),
            'adv_eval_confusion_matrix': encode_array(adv_eval_cm),
        })

    return result


def sigma_grid_evaluation(sigmas, n=300):
    """Decoupled Gaussian robustness evaluation on a fixed sigma-grid.

    Unlike ``visual_evaluation`` (which reads the producer's adversarial eval
    stream, whose noise level equals the vehicle's training-augmentation knob
    Mp_std/Bp_std), this evaluation samples the CLEAN observation buffers and
    injects zero-mean Gaussian noise of std=sigma into the two brake-pressure
    features (usBpPres/usMpPres) of the ANOMALY and ATTACK samples for each
    sigma in ``sigmas``. Normal samples are left clean, matching the
    Mp_std/Bp_std threat model.

    Because the grid is a fixed configuration constant and the source samples
    are clean, the eval target is fully decoupled from how the model was
    trained and is identical across the whole fleet by construction — which is
    exactly what the ET2/ET3 comparisons require. Results are logged under
    ``adv_eval/sigma_<s>/*`` so each sigma forms its own W&B sub-section.

    Returns None if the clean buffers are not yet warm enough.
    """
    global brain
    per_class = max(n // 3, 1)
    diag_feats, diag_labels = diagnostics_buffer.sample(per_class)
    anom_feats, anom_labels = anomalies_buffer.sample(per_class)
    atk_feats, atk_labels = attacks_buffer.sample(per_class)

    if len(diag_feats) < 10 or len(anom_feats) < 10 or len(atk_feats) < 10:
        return None

    # Only anomaly/attack pressures are perturbed; normal samples stay clean.
    pert_feats = torch.vstack((anom_feats, atk_feats))
    y = torch.vstack((diag_labels, anom_labels, atk_labels)).numpy().ravel()

    result = {}
    for sigma in sigmas:
        sigma = float(sigma)
        noisy = pert_feats.clone()
        noise = torch.randn(noisy.shape[0], len(HSJA_PRESSURE_FEATURE_INDICES)) * sigma
        noisy[:, HSJA_PRESSURE_FEATURE_INDICES] += noise
        feats = torch.vstack((diag_feats, noisy))

        brain.model.eval()
        with brain.model_lock, torch.no_grad():
            preds, _ = brain.model(feats)
            preds = preds.argmax(dim=1).numpy()

        tag = f"{sigma:g}".replace('.', '_')
        result[f'adv_eval/sigma_{tag}/accuracy']  = accuracy_score(y, preds)
        result[f'adv_eval/sigma_{tag}/precision'] = precision_score(y, preds, zero_division=0, average='weighted')
        result[f'adv_eval/sigma_{tag}/recall']    = recall_score(y, preds, zero_division=0, average='weighted')
        result[f'adv_eval/sigma_{tag}/f1']        = f1_score(y, preds, zero_division=0, average='weighted')
        result[f'adv_eval/sigma_{tag}/macro_f1']  = f1_score(y, preds, zero_division=0, average='macro')

    return result


def hsja_evaluation(n_per_class=10, n_steps=30, n_grad_samples=30, include_plots=True,
                    feature_indices=None, clean_anchors=True):
    """
    Run HopSkipJump attack on a small sample from each class buffer.

    For each sample the attacker has only hard-label (decision) access to the
    local classifier, matching a realistic black-box threat model.
    Returns None if buffers are not yet warm enough; otherwise reports metrics
    directly via metrics_reporter and clears _hsja_eval_running.

    The '/' separator in scalar keys (e.g. 'hsja_adv_eval/accuracy') creates
    a dedicated W&B sub-section distinct from the Gaussian 'adv_eval_*' group.
    """
    global brain, _hsja_eval_running

    logger.info(
        f"HSJA eval round STARTING — n_per_class={n_per_class}, n_steps={n_steps}, "
        f"n_grad_samples={n_grad_samples}, include_plots={include_plots}, "
        f"clean_anchors={clean_anchors}, "
        f"feature_indices={'all' if feature_indices is None else feature_indices}."
    )

    # With clean_anchors=True the attack starts from clean samples for every
    # vehicle, so HSJA measures the model's decision boundary rather than the
    # noise level baked into each vehicle's producer eval stream. This makes
    # avg_perturbation / avg_queries directly comparable across the fleet
    # (the default, decoupled behaviour). Set clean_anchors=False to recover
    # the legacy behaviour of attacking the noisy producer eval stream.
    anom_buf = anomalies_buffer if clean_anchors else eval_anomalies_buffer
    atk_buf = attacks_buffer if clean_anchors else eval_attacks_buffer

    diag_feats, diag_labels = diagnostics_buffer.sample(n_per_class)
    anom_feats, anom_labels = anom_buf.sample(n_per_class)
    atk_feats, atk_labels = atk_buf.sample(n_per_class)

    if len(diag_feats) < 5 or len(anom_feats) < 5 or len(atk_feats) < 5:
        # Silent omission guard: the round was triggered but cannot run because
        # one or more class buffers are not warm enough yet. Make it explicit so
        # missing HSJA rounds are not mistaken for a crash.
        logger.warning(
            f"HSJA eval round OMITTED — buffers not warm enough "
            f"(need >=5 each; have diagnostics={len(diag_feats)}, "
            f"anomalies={len(anom_feats)}, attacks={len(atk_feats)}). "
            f"Will retry on the next benchmark trigger."
        )
        _hsja_eval_running = False
        return

    all_feats = torch.vstack((diag_feats, anom_feats, atk_feats))
    all_labels_arr = torch.vstack((diag_labels, anom_labels, atk_labels)).squeeze(1).numpy()  # (N,)

    # predict_fn acquires model_lock per query — short critical sections that
    # do not starve the training thread. inference_mode is used instead of
    # no_grad: it skips view/version tracking entirely for a cheaper forward.
    def predict_fn(x: torch.Tensor) -> int:
        brain.model.eval()
        with brain.model_lock, torch.inference_mode():
            logits, _ = brain.model(x.unsqueeze(0))
            return logits.argmax(dim=-1).item()

    # Batched variant: evaluates k probes in a single forward pass / single
    # lock acquisition. Returns plain Python ints so no inference-mode tensor
    # escapes the critical section. Used by the gradient-estimation phase,
    # where the probes are mutually independent.
    def predict_batch_fn(X: torch.Tensor) -> list:
        brain.model.eval()
        with brain.model_lock, torch.inference_mode():
            logits, _ = brain.model(X)
            return logits.argmax(dim=-1).tolist()

    adv_examples = []
    adv_preds = []
    pert_norms = []
    total_queries = 0

    n_samples = len(all_feats)
    round_t0 = time.time()
    for i in range(n_samples):
        if stop_threads:
            logger.warning(
                f"HSJA eval round ABORTED — shutdown requested mid-evaluation "
                f"(processed {i}/{n_samples} samples)."
            )
            _hsja_eval_running = False
            return
        x = all_feats[i]
        y_i = int(all_labels_arr[i])
        sample_t0 = time.time()
        x_adv, n_q = hopskipjump_attack(
            predict_fn, x, y_i,
            n_steps=n_steps,
            n_grad_samples=n_grad_samples,
            feature_indices=feature_indices,
            predict_batch_fn=predict_batch_fn,
        )
        total_queries += n_q
        pert_norms.append(float(torch.norm(x_adv - x)))
        adv_examples.append(x_adv)
        with brain.model_lock, torch.inference_mode():
            brain.model.eval()
            adv_logits, _ = brain.model(x_adv.unsqueeze(0))
            adv_preds.append(adv_logits.argmax(dim=-1).item())
        # Per-sample progress so a slow/stuck architecture is visible live.
        sample_dt = time.time() - sample_t0
        logger.info(
            f"HSJA progress: sample {i + 1}/{n_samples} done "
            f"(true_class={y_i}, queries={n_q}, {sample_dt:.2f}s); "
            f"cumulative_queries={total_queries}, "
            f"elapsed={time.time() - round_t0:.1f}s."
        )

    adv_preds_arr = np.array(adv_preds)

    adv_accuracy  = accuracy_score(all_labels_arr, adv_preds_arr)
    adv_precision = precision_score(all_labels_arr, adv_preds_arr, zero_division=0, average='weighted')
    adv_recall    = recall_score(all_labels_arr, adv_preds_arr, zero_division=0, average='weighted')
    adv_f1        = f1_score(all_labels_arr, adv_preds_arr, zero_division=0, average='weighted')
    adv_macro_f1  = f1_score(all_labels_arr, adv_preds_arr, zero_division=0, average='macro')

    result = {
        'hsja_adv_eval/accuracy':         adv_accuracy,
        'hsja_adv_eval/precision':        adv_precision,
        'hsja_adv_eval/recall':           adv_recall,
        'hsja_adv_eval/f1':               adv_f1,
        'hsja_adv_eval/macro_f1':         adv_macro_f1,
        'hsja_adv_eval/avg_perturbation': float(np.mean(pert_norms)),
        'hsja_adv_eval/avg_queries':      total_queries / max(len(all_feats), 1),
    }

    if include_plots:
        adv_cm = confusion_matrix(all_labels_arr, adv_preds_arr, labels=[0, 1, 2])

        # PCA of original (clean) feature space for the left panel
        X = all_feats - all_feats.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(X, q=2)
        X2 = (X @ V[:, :2]).numpy()

        # Manifold coordinates of adversarial examples for the centre/right panels
        adv_stack = torch.stack(adv_examples)
        with brain.model_lock, torch.inference_mode():
            brain.model.eval()
            _, adv_manifold = brain.model(adv_stack)
        adv_manifold = adv_manifold.numpy()

        result.update({
            'hsja_visual_eval_X':        encode_array(X2),
            'hsja_visual_eval_y':        encode_array(all_labels_arr),
            'hsja_visual_eval_preds':    encode_array(adv_preds_arr),
            'hsja_visual_eval_manifold': encode_array(adv_manifold),
            'hsja_adv_eval_confusion_matrix': encode_array(adv_cm),
        })

    logger.info(
        f"HSJA eval done — accuracy={adv_accuracy:.3f}, "
        f"avg_perturbation={result['hsja_adv_eval/avg_perturbation']:.4f}, "
        f"avg_queries={result['hsja_adv_eval/avg_queries']:.0f}, "
        f"total_queries={total_queries}, "
        f"round_time={time.time() - round_t0:.1f}s ({n_samples} samples)"
    )

    metrics_reporter.report(result)
    _hsja_eval_running = False


def plot_results(Y, all_preds, pca_embed, manifold, task_name):

        _, axes = plt.subplots(1, 3, figsize=(20, 4))

        colors = ['r', 'g', 'b']

        ax = axes[0]
        for eventype in EventType:
            mask = Y.squeeze() == eventype.value
            ax.scatter(pca_embed[mask, 0], pca_embed[mask, 1],
                    c=colors[eventype.value], s=15, alpha=0.1, label=eventype.name)
        ax.set_title(f'Input-Space (2D-PCA) {task_name}')
        ax.legend()

        ax = axes[1]
        for eventype in EventType:
            mask = Y.squeeze() == eventype.value
            ax.scatter(manifold[mask, 0], manifold[mask, 1],
                    c=colors[eventype.value], s=15, alpha=0.1, label=eventype.name)
        ax.set_title(f'2D-Representation-Space (labels) {task_name}')
        ax.legend()

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
    conf_cons = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': f'{VEHICLE_NAME}-consumer-group'+generate_random_string(7),
        'auto.offset.reset': 'earliest'
    }
    return Consumer(conf_cons)


def check_and_create_topics(topic_list):
    """
    Check if the specified topics exist in Kafka, and create them if missing.
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


def delete_owned_topics():
    """Delete topics owned by this consumer container.
    Called on shutdown so experiments always start from a clean Kafka state."""
    owned = [
        f"{VEHICLE_NAME}_statistics",
        f"{VEHICLE_NAME}_weights",
    ]
    try:
        admin = AdminClient({'bootstrap.servers': KAFKA_BROKER})
        futures = admin.delete_topics(owned, operation_timeout=10)
        for topic, future in futures.items():
            try:
                future.result()
                logger.info(f"Deleted Kafka topic: {topic}")
            except Exception as e:
                logger.warning(f"Could not delete topic {topic} (may not exist or Kafka down): {e}")
    except Exception as e:
        logger.warning(f"Topic deletion failed (Kafka may be down): {e}")


def deserialize_message(msg):
    try:
        message_value = json.loads(msg.value().decode('utf-8'))
        logger.debug(f"received message from topic [{msg.topic()}]")
        return message_value
    except json.JSONDecodeError as e:
        logger.error(f"Error deserializing message: {e}")
        return None


def process_message(topic, msg):
    global anomalies_buffer, diagnostics_buffer, attacks_buffer
    global eval_anomalies_buffer, eval_attacks_buffer
    global anoms_processed, diagnostics_processed, attacks_processed, records_processed
    global eval_anomalies_processed, eval_attacks_processed

    counting_message = False

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
            anoms_processed += 1

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
        logger.info(f"Received {records_processed} messages: {attacks_processed} attacks, {anoms_processed} anomalies, {diagnostics_processed} diagnostics.")
        logger.info(f"Received {eval_anomalies_processed} eval_anomalies, {eval_attacks_processed} eval_attacks.")


def send_attack_mitigation_request(vehicle_name):
    global mitigation_times, lists_lock

    url = f"http://{MANAGER_IP}:{MANAGER_PORT}/stop-attack"
    data = {"vehicle_name": vehicle_name, "origin": "AI"}
    response = requests.post(url, json=data)
    try:
        response_json = response.json()
        # logger.info(f"Mitigate-attack Response JSON: {response_json}")
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
    url = f"http://{MANAGER_IP}:{MANAGER_PORT}/vehicle-status"
    data = {"vehicle_name": vehicle_name}
    response = requests.post(url, json=data)
    logger.debug(f"Vehicle-status Response Status Code: {response.status_code}")
    logger.debug(f"Vehicle-status Response Body: {response.text}")
    return response.text


def mitigation_and_rewarding(prediction, current_label):
    global mitigation_reward
    if prediction == 2:
        if current_label == prediction:
            if MITIGATION:
                if get_status_from_manager(VEHICLE_NAME) == "INFECTED":
                    send_attack_mitigation_request(VEHICLE_NAME)
            mitigation_reward += true_positive_reward
        else:
            mitigation_reward += false_positive_reward
    else:
        if current_label == prediction:
            mitigation_reward += true_negative_reward
        else:
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
    global consumer
    topics = [f"{VEHICLE_NAME}_anomalies", f"{VEHICLE_NAME}_eval_anomalies", f"{VEHICLE_NAME}_normal_data"]
    consumer.subscribe(topics)
    global_weights_puller.subscribe()
    logger.debug(f"(re)subscribed to topics: {topics}")


def consume_vehicle_data():
    global consumer

    stats_topic = f"{VEHICLE_NAME}_statistics"
    weights_topic = f"{VEHICLE_NAME}_weights"

    check_and_create_topics([stats_topic, weights_topic])

    consumer = create_consumer()
    subscribe_to_topics()

    try:
        while not stop_threads:
            msg = consumer.poll(5.0)
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
            # Update the FedProx anchor point to the freshly received global model.
            # When fedprox_mu == 0 this is a no-op (the stored reference is never read).
            brain.set_global_reference(new_weights)
            logger.info("Local weights updated using global model.")


def _run_hsja_evaluation_bg(**kwargs):
    """Background thread target: run HSJA evaluation and ensure the running flag is cleared."""
    global _hsja_eval_running
    try:
        hsja_evaluation(
            n_per_class=kwargs.get('hsja_n_per_class', 10),
            n_steps=kwargs.get('hsja_n_steps', 30),
            n_grad_samples=kwargs.get('hsja_n_grad_samples', 30),
            include_plots=kwargs.get('include_plots', True),
            feature_indices=kwargs.get('hsja_feature_indices', None),
            clean_anchors=kwargs.get('hsja_clean_anchors', True),
        )
    except Exception as e:
        logger.error(f"HSJA evaluation raised an exception: {e}", exc_info=True)
        _hsja_eval_running = False
    finally:
        # Always confirm the background thread has exited, regardless of whether
        # the round completed, was omitted (cold buffers), aborted, or errored.
        logger.info("HSJA evaluation background thread exiting.")


def train_model(**kwargs):
    global brain, batch_counter, epoch_counter
    global epoch_loss
    global epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_macro_f1
    global mitigation_reward, mitigation_times
    global online_batch_labels, online_main_batch_preds
    global lists_lock
    global anoms_processed, diagnostics_processed, attacks_processed, records_processed
    global eval_anomalies_processed, eval_attacks_processed
    global _hsja_eval_running, _hsja_eval_thread
    global benchmark_eval_counter
    lists_lock = Lock()

    batch_size = kwargs.get('batch_size', 32)
    epoch_size = kwargs.get('epoch_size', 50)
    save_model_freq_epochs = kwargs.get('save_model_freq_epochs', 10)
    run_benchmarks_freq_epochs = kwargs.get('run_benchmarks_freq_epochs', save_model_freq_epochs)
    plot_creation_freq_benchmarks = kwargs.get('plot_creation_freq_benchmarks', 3)
    hsja_enabled = kwargs.get('hsja_enabled', True)
    eval_sigmas = kwargs.get('eval_sigmas', [0.5, 1.0, 1.5, 2.0])

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

            if adversarial_training and len(adv_anomalies_feats) >= batch_size and len(adv_attack_feats) >= batch_size:
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
            batch_macro_f1 = f1_score(batch_main_labels, batch_main_preds, zero_division=0, average='macro')

            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            epoch_precision += batch_precision
            epoch_recall += batch_recall
            epoch_f1 += batch_f1
            epoch_macro_f1 += batch_macro_f1

            if batch_counter % epoch_size == 0:

                epoch_counter += 1
                epoch_loss /= epoch_size
                epoch_accuracy /= epoch_size
                epoch_precision /= epoch_size
                epoch_recall /= epoch_size
                epoch_f1 /= epoch_size
                epoch_macro_f1 /= epoch_size

                metrics_dict = {
                    'total_loss': epoch_loss,
                    'class_accuracy': epoch_accuracy,
                    'class_precision': epoch_precision,
                    'class_recall': epoch_recall,
                    'class_f1': epoch_f1,
                    'class_macro_f1': epoch_macro_f1,
                    'diagnostics_processed': diagnostics_processed,
                    'anoms_processed': anoms_processed,
                    'attacks_processed': attacks_processed,
                    'records_processed': records_processed,
                    'eval_anoms_processed': eval_anomalies_processed,
                    'eval_attacks_processed': eval_attacks_processed
                }

                if len(online_batch_labels) > 20:
                    with lists_lock:
                        online_main_batch_accuracy = accuracy_score(online_batch_labels, online_main_batch_preds)
                        online_main_batch_precision = precision_score(online_batch_labels, online_main_batch_preds, zero_division=0, average='weighted')
                        online_main_batch_recall = recall_score(online_batch_labels, online_main_batch_preds, zero_division=0, average='weighted')
                        online_main_batch_f1 = f1_score(online_batch_labels, online_main_batch_preds, zero_division=0, average='weighted')
                        online_main_batch_macro_f1 = f1_score(online_batch_labels, online_main_batch_preds, zero_division=0, average='macro')
                        online_cm = confusion_matrix(online_batch_labels, online_main_batch_preds, labels=[0, 1, 2])

                        online_metrics_dict = {
                            'online_class_accuracy': online_main_batch_accuracy,
                            'online_class_precision': online_main_batch_precision,
                            'online_class_recall': online_main_batch_recall,
                            'online_class_f1': online_main_batch_f1,
                            'online_class_macro_f1': online_main_batch_macro_f1,
                            'online_confusion_matrix': encode_array(online_cm),
                            }

                        online_metrics_dict['mitigation_time'] = np.array(mitigation_times).mean() if len(mitigation_times) > 0 else 0.0
                        online_metrics_dict['mitigation_reward'] = mitigation_reward

                        metrics_dict.update(online_metrics_dict)

                        online_batch_labels = []
                        online_main_batch_preds = []
                        mitigation_times = []
                        mitigation_reward = 0

                metrics_reporter.report(metrics_dict)
                epoch_loss = epoch_accuracy = epoch_precision = epoch_recall = epoch_f1 = epoch_macro_f1 = 0

                if epoch_counter % save_model_freq_epochs == 0:
                    model_path = kwargs.get('model_saving_path', 'default_model.pth')
                    logger.info(f"Saving model after {epoch_counter} epochs as {model_path}.")
                    brain.save_model()

                if epoch_counter % run_benchmarks_freq_epochs == 0:
                    benchmark_eval_counter += 1
                    include_plots = (benchmark_eval_counter % plot_creation_freq_benchmarks == 0)

                    visual_eval_dict = visual_evaluation(include_plots=include_plots)
                    if visual_eval_dict is not None:
                        logger.info(f"Sending visual evaluation results to wandber (plots={include_plots})...")
                        metrics_reporter.report(visual_eval_dict)

                    # Decoupled, fleet-identical Gaussian robustness curve.
                    if eval_sigmas:
                        sigma_eval_dict = sigma_grid_evaluation(eval_sigmas)
                        if sigma_eval_dict is not None:
                            logger.info(f"Sending sigma-grid evaluation results to wandber (sigmas={eval_sigmas})...")
                            metrics_reporter.report(sigma_eval_dict)

                    # Trigger HSJA evaluation in a background thread so it does not
                    # block the training loop. Skipped if the previous run is ongoing.
                    if hsja_enabled and not _hsja_eval_running:
                        _hsja_eval_running = True
                        _hsja_eval_thread = threading.Thread(
                            target=_run_hsja_evaluation_bg,
                            kwargs={**kwargs, 'include_plots': include_plots},
                            daemon=True
                        )
                        _hsja_eval_thread.start()
                        logger.info(
                            f"HSJA evaluation thread started "
                            f"(benchmark round {benchmark_eval_counter}, epoch {epoch_counter})."
                        )
                    elif hsja_enabled and _hsja_eval_running:
                        # The previous round has not finished yet, so this trigger
                        # is skipped. Logged at WARNING so repeated/overlapping
                        # omissions are visible rather than silently dropped.
                        logger.warning(
                            f"HSJA evaluation OMITTED for benchmark round "
                            f"{benchmark_eval_counter} (epoch {epoch_counter}): "
                            f"previous round still running."
                        )
                    elif not hsja_enabled:
                        logger.info(
                            f"HSJA evaluation OMITTED for benchmark round "
                            f"{benchmark_eval_counter} (epoch {epoch_counter}): "
                            f"disabled by config (hsja_enabled=False)."
                        )

        time.sleep(kwargs.get('training_freq_seconds', 1))


def signal_handler(sig, frame):
    global stop_threads, stats_consuming_thread, training_thread, pushing_weights_thread, pulling_weights_thread
    logger.debug(f"Received signal {sig}. Gracefully stopping {VEHICLE_NAME} producer.")
    stop_threads = True


def resubscribe():
    while not stop_threads:
        try:
            time.sleep(resubscribe_interval_seconds)
            subscribe_to_topics()
        except Exception as e:
            logger.error(f"Error in periodic resubscription: {e}")


def parse_str_list(arg):
    try:
        return [str(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Arguments must be strings separated by commas")


def configure_no_proxy():
    os.environ['no_proxy'] = os.environ.get('no_proxy', '') + f",{HOST_IP},{MANAGER_IP}"


def start_consumer_runtime(args_namespace):
    global VEHICLE_NAME, KAFKA_BROKER, MANAGER_PORT, MANAGER_IP, MITIGATION, mode, average_param
    global batch_size, stop_threads, stats_consuming_thread, training_thread, pushing_weights_thread, pulling_weights_thread
    global attacks_buffer, anomalies_buffer, diagnostics_buffer, brain, metrics_reporter, logger, weights_reporter, global_weights_puller
    global eval_attacks_buffer, eval_anomalies_buffer, adversarial_training
    global resubscribe_interval_seconds, epoch_batches
    global true_positive_reward, false_positive_reward, true_negative_reward, false_negative_reward
    global batch_counter, epoch_counter, records_processed, attacks_processed, anoms_processed
    global diagnostics_processed, eval_anomalies_processed, eval_attacks_processed
    global epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_macro_f1
    global online_batch_labels, online_main_batch_preds, mitigation_times, mitigation_reward
    global _hsja_eval_running, _hsja_eval_thread

    # Reset all per-run accumulators so each run (within a reused container)
    # starts from a clean slate, matching the fresh W&B run's step 0.
    batch_counter = 0
    epoch_counter = 0
    records_processed = 0
    attacks_processed = 0
    anoms_processed = 0
    diagnostics_processed = 0
    eval_anomalies_processed = 0
    eval_attacks_processed = 0
    epoch_loss = 0
    epoch_accuracy = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_macro_f1 = 0
    online_batch_labels = []
    online_main_batch_preds = []
    mitigation_times = []
    mitigation_reward = 0
    _hsja_eval_running = False
    _hsja_eval_thread = None

    args = args_namespace

    MITIGATION = args.mitigation
    MANAGER_PORT = args.manager_port
    MANAGER_IP = args.manager_ip

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

    logger.info(f"Starting consumer for vehicle {VEHICLE_NAME}")
    logger.info(f"Adversarial training: {args.adversarial_training}")
    logger.info("All arguments:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    adversarial_training = args.adversarial_training

    logger.info(f"Starting brain for vehicle {VEHICLE_NAME}")
    if args.seed is not None:
        logger.info(f"Random torch seed will be set to {args.seed}")
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
    try:
        metrics_reporter.producer.flush(5)
        weights_reporter.producer.flush(5)
        logger.info("Kafka producers flushed.")
    except Exception as e:
        logger.error(f"Error flushing Kafka producers: {e}")
    delete_owned_topics()
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
