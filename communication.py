from confluent_kafka import SerializingProducer, Consumer, KafkaError
from confluent_kafka.serialization import StringSerializer
import json
import logging
import pickle

from OpenFAIR.packet_loss import PacketLossSimulator
from OpenFAIR.network_delay import NetworkDelaySimulator

class WeightsReporter:
    def __init__(self, **kwargs):
        kafka_broker_url = kwargs.get('kafka_broker')
        self.vehicle_name = kwargs.get('vehicle_name')
        conf_prod_weights={
        'bootstrap.servers': kafka_broker_url,  # Kafka broker URL
        'key.serializer': StringSerializer('utf_8'),
        'value.serializer': lambda v, ctx: pickle.dumps(v)
         }
        self.producer = SerializingProducer(conf_prod_weights)
        self.packet_loss = PacketLossSimulator(kwargs.get('packet_loss_rate', 0.1))
        # Simulated latency+jitter on the {vehicle}_weights upload to the FL
        # manager (same policy as packet_loss — {vehicle}_statistics is
        # W&B-bound and never delayed, see MetricsReporter).
        self.network_delay = NetworkDelaySimulator(
            kwargs.get('delay_mean_ms', 0.0), kwargs.get('jitter_std_ms', 0.0))

        self.logger = logging.getLogger("weights_upload_" + kwargs['vehicle_name'])
        self.logger.setLevel(str(kwargs.get('logging_level', 'INFO')).upper())


    def push_weights(self, weights):
        weights_topic=f"{self.vehicle_name}_weights"
        if self.packet_loss.should_drop():
            self.logger.debug(f"[packet-loss] dropped weights update for topic {weights_topic} "
                              f"(rate={self.packet_loss.packet_loss_rate})")
            return

        def _deliver():
            try:
                self.producer.produce(topic=weights_topic, value=weights)
                self.producer.flush()
                self.logger.info(f"Published to topic: {weights_topic}")
            except Exception as e:
                self.logger.error(f"Failed to produce weights: {e}")

        self.network_delay.send(_deliver)


class MetricsReporter:
    # Publishes to {vehicle}_statistics, which Wandber subscribes to directly
    # ('^.*_statistics$') for W&B logging (inference/robustness metrics,
    # mitigation reward, plots, ...). Never subject to simulated packet loss —
    # only the producer -> consumer telemetry / consumer -> FL weights are.
    def __init__(self, **kwargs):
        kafka_broker_url = kwargs.get('kafka_broker')
        self.vehicle_name = kwargs.get('vehicle_name')
        conf_prod_stat={
        'bootstrap.servers': kafka_broker_url,  # Kafka broker URL
        'key.serializer': StringSerializer('utf_8'),
        'value.serializer': lambda v, ctx: json.dumps(v)
         }

        self.producer = SerializingProducer(conf_prod_stat)
        self.logger = logging.getLogger("metrics_reporter_" + kwargs['vehicle_name'])
        self.logger.setLevel(str(kwargs.get('logging_level', 'INFO')).upper())

    def report(self, metrics):

        stats = {
            'vehicle_name' : self.vehicle_name,
        }
        stats.update(metrics)

        topic_statistics=f"{self.vehicle_name}_statistics"
        try:
            self.producer.produce(topic=topic_statistics, value=stats)
            self.producer.flush()
            self.logger.debug(f"Published to topic: {topic_statistics}")
        except Exception as e:
            self.logger.error(f"Failed to produce statistics: {e}")


class WeightsPuller:

    def __init__(self, **kwargs):
        kafka_broker_url = kwargs.get('kafka_broker')
        self.vehicle_name = kwargs.get('vehicle_name')
        self.consumer = Consumer({
            'bootstrap.servers': kafka_broker_url,  # Kafka broker URL
            'group.id': f'{self.vehicle_name}-consumer-group',  # Consumer group ID for message offset tracking
            'auto.offset.reset': 'earliest'  # Start reading from the earliest message if no offset is present
        })
        self.subscribe()
        self.logger = logging.getLogger("glob_weight_puller" + kwargs['vehicle_name'])
        self.logger.setLevel(str(kwargs.get('logging_level', 'INFO')).upper())


    def subscribe(self):
        try:
            self.consumer.subscribe(["global_weights"])
        except Exception as e:
            self.logger.error(f"Failed to subscribe to global weights topic: {e}")

    def pull_weights(self):
        # self.logger.debug("Pulling global weights")
        weights = None
        try:
            msg = self.consumer.poll(timeout=10.0)
            if msg is None:
                self.logger.info("No new global weights received.")
            elif not msg.error():
                weights = pickle.loads(msg.value())
                self.logger.info(f"Received new global weights")
            elif msg.error().code() != KafkaError._PARTITION_EOF:
                self.logger.error(f"Error while consuming weights: {msg.error()}")
        except Exception as e:
            self.logger.error(f"Failed to consume weights: {e}")
        return weights


    def close(self):
        self.consumer.close()