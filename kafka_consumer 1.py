from confluent_kafka import Consumer, KafkaError
import json

# Kafka consumer configuration
conf = {
    'bootstrap.servers': 'localhost:9092',  # Replace with your Kafka broker(s)
    'group.id': 'python-consumer',
    'auto.offset.reset': 'earliest'
}

# Create Kafka consumer
consumer = Consumer(conf)

# Subscribe to topics
topics = ["leejam","recognition_topic", "area_topic"]
consumer.subscribe(topics)

try:
    while True:
        # Poll for new messages
        message = consumer.poll(timeout=1.0)

        if message is None:
            continue
        if message.error():
            if message.error().code() == KafkaError._PARTITION_EOF:
                # End of partition, the consumer reached the end of the partition
                print(f"Reached end of partition {message.partition()}")
            elif message.error():
                # Error, handle the error
                print(f"Error: {message.error()}")
        else:
            # Message was successfully received
            message_value = json.loads(message.value().decode('utf-8'))
            print(f"Received message: {message_value}")

except KeyboardInterrupt:
    # User interrupted the consumer with Ctrl+C
    pass

finally:
    # Close the consumer
    consumer.close()
