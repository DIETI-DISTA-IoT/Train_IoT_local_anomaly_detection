from confluent_kafka import SerializingProducer, Consumer, KafkaError
from confluent_kafka.serialization import StringSerializer
import json
import logging
import pickle

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

        self.logger = logging.getLogger("weights_upload_" + kwargs['vehicle_name'])
        self.logger.setLevel(str(kwargs.get('logging_level', 'INFO')).upper())
    

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