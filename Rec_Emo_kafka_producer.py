import numpy as np
import cv2
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from keras.models import load_model
import numpy as np
from PIL import Image
from keras.preprocessing import image
from retinaface import RetinaFace
from confluent_kafka import Producer
import json
import time
import random

# Kafka producer configuration
conf = {
    'bootstrap.servers': 'localhost:9092',  # Replace with your Kafka broker(s)
    'client.id': 'python-producer'
}

# Create Kafka producer
producer = Producer(conf)

# Function to send message to Kafka topic
def send_message(topic, message):
    producer.produce(topic, value=json.dumps(message))
    producer.flush()

# Sample data for demonstration
cam_id = "camera-1"
timestamp = str(time.time())


# Load pre-trained Facenet model
model = FaceNet()

# Load emotion detection model
emotion_model_path = 'emotion.hdf5'
emotion_classifier = load_model(emotion_model_path)

def display_emotion(number):
    emotions = {0: 'Negative', 1: 'Negative', 2: 'Negative', 3: 'Positive',
                4: 'Neutral', 5: 'Positive', 6: 'Neutral'}
    return emotions[number]

# Load all embeddings from the NPZ files
known_embeddings = {}
known_names = {}

# Load Anjay embeddings
data = np.load('employee_embeddings.npz')
for arr_name in data.files:
    known_embeddings[arr_name] = data[arr_name]
    known_names[arr_name] = arr_name


# Load RetinaFace model
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
retinaface_face_detection_path = 'det_10g.onnx'
retinaface_det_model = RetinaFace(retinaface_face_detection_path, providers=PROVIDERS)
retinaface_det_model.prepare(ctx_id=1, input_size=(640, 640), det_thresh=0.7)

# Open video file
video_path = 'SideDesk2.mp4'
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('FRandEmotionoutput4min.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}/{total_frames}")

    # Start timer
    # start_time = time.time()

    # Extract faces using RetinaFace
    faces, _ = retinaface_det_model.detect(frame,max_num=0,metric='default')

    for face in faces:
        
        x1, y1, x2, y2, _ = face
        face_img = frame[int(y1):int(y2), int(x1):int(x2)]

        # Convert frame to RGB (facenet expects RGB images)
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Extract embeddings for the face
        input_embeddings = model.extract(rgb_face, threshold=0.7)

        for input_embedding in input_embeddings:
            input_embedding_vector = input_embedding["embedding"].reshape(1, -1)
            recognized_name = "Unknown"

            # Compare the input embedding with known embeddings
            max_similarity = 0
            for name, embeddings in known_embeddings.items():
                for emb in embeddings:
                    similarity = cosine_similarity(input_embedding_vector, emb.reshape(1, -1))
                    if similarity > max_similarity:
                        max_similarity = similarity
                        recognized_name = name

            if max_similarity > 0.7:  # Adjust the threshold as needed
                print(f"Recognized: {recognized_name} (Similarity: {max_similarity})")

            # Perform emotion detection
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (64, 64))  # Resize to match emotion detection model input size
            gray_face = gray_face.astype('float32') / 255.0
            gray_face = np.expand_dims(gray_face, axis=0)
            gray_face = np.expand_dims(gray_face, axis=-1)

            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion = display_emotion(emotion_label_arg)

            text = recognized_name + f" | {emotion}"

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = int(x1) + int((x2 - x1 - text_size[0]) / 2)
            text_y = int(y1) - 10
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

            EmployeeEmotionData = {
    'employeeId': 123,
    'employeeName': recognized_name,
    'cameraId': cam_id,
    'emotion': emotion,
    'timestamp': timestamp,
  }
            send_message("leejam", EmployeeEmotionData)

    # Stop timer
    # end_time = time.time()

    # Calculate time taken per frame
    # time_per_frame = end_time - start_time
    # print(f"Time taken per frame: {time_per_frame:.4f} seconds")

    # Write frame to output video
    output_video.write(frame)

    # Display frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
output_video.release()
cv2.destroyAllWindows()
