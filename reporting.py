from confluent_kafka import Consumer, KafkaError, KafkaException, Producer, SerializingProducer
import os
from confluent_kafka.serialization import StringSerializer
import json
import logging


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
        logging_level = kwargs.get('logging_level', 'INFO')
    
        self.logger = logging.getLogger("REPORTER_" + kwargs['container_name'])
        self.logger.setLevel(kwargs.get('logging_level', 'INFO'))

    def report(self, metrics):

        stats = {
            'vehicle_name' : self.vehicle_name,
        }
        stats.update(metrics)

        topic_statistics=f"{self.vehicle_name}_statistics"
        try:
            self.producer.produce(topic=topic_statistics, value=stats)
            self.producer.flush()
            self.logger.debug(f"{self.vehicle_name}_BRAIN: published to topic: {topic_statistics}")
        except Exception as e:
            self.logger.error(f"{self.vehicle_name}_BRAIN: Failed to produce statistics: {e}")
