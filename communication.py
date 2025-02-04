from confluent_kafka import SerializingProducer, Consumer, KafkaError
from confluent_kafka.serialization import StringSerializer
import json
import logging
import pickle
import io
import torch

class WeightsReporter:
    def __init__(self, **kwargs):
        kafka_broker_url = kwargs.get('kafka_broker')
        self.vehicle_name = kwargs.get('vehicle_name')
        conf_prod_weights={
        'bootstrap.servers': kafka_broker_url,  # Kafka broker URL
        'value.serializer': self._serialize_state_dict
         }
        self.producer = SerializingProducer(conf_prod_weights)

        self.logger = logging.getLogger("weights_upload_" + kwargs['container_name'])
        self.logger.setLevel(kwargs.get('logging_level', str(kwargs.get('logging_level', 'INFO')).upper()))
    

    def _serialize_state_dict(self, state_dict, ctx):
        """Serialize PyTorch state dict to bytes using torch.save"""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    def push_weights(self, weights):
        weights_topic=f"{self.vehicle_name}_weights"
        try:
            self.producer.produce(topic=weights_topic, value=weights)
            self.producer.flush()
            self.logger.info(f"Published to topic: {weights_topic}")
        except Exception as e:
            self.logger.error(f"Failed to produce weights: {e}")


class MetricsReporter:
    def __init__(self, **kwargs):
        kafka_broker_url = kwargs.get('kafka_broker')
        self.vehicle_name = kwargs.get('vehicle_name')
        conf_prod_stat={
        'bootstrap.servers': kafka_broker_url,  # Kafka broker URL
        'key.serializer': StringSerializer('utf_8'),
        'value.serializer': lambda v, ctx: json.dumps(v)
         }
        
        self.producer = SerializingProducer(conf_prod_stat)    
        self.logger = logging.getLogger("metrics_reporter_" + kwargs['container_name'])
        self.logger.setLevel(kwargs.get('logging_level', str(kwargs.get('logging_level', 'INFO')).upper()))

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
        self.logger = logging.getLogger("glob_weight_puller" + kwargs['container_name'])
        self.logger.setLevel(kwargs.get('logging_level', str(kwargs.get('logging_level', 'INFO')).upper()))


    def subscribe(self):
        try:
            self.consumer.subscribe(["global_weights"])
        except Exception as e:
            self.logger.error(f"Failed to subscribe to global weights topic: {e}")


    @staticmethod
    def deserialize_state_dict(bytes_data):
        """Deserialize bytes back to PyTorch state dict"""
        buffer = io.BytesIO(bytes_data)
        return torch.load(buffer)
    

    def pull_weights(self):
        # self.logger.debug("Pulling global weights")
        weights = None
        try:
            msg = self.consumer.poll(timeout=10.0)
            if msg is None:
                self.logger.info("No new global weights received.")
            elif not msg.error():
                weights = self.deserialize_state_dict(msg.value())
                self.logger.info(f"Received new global weights")
            elif msg.error().code() != KafkaError._PARTITION_EOF:
                self.logger.error(f"Error while consuming weights: {msg.error()}")
        except Exception as e:
            self.logger.error(f"Failed to consume weights: {e}")
        return weights


    def close(self):
        self.consumer.close()