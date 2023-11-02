import numpy as np
import cv2
from keras.preprocessing import image_dataset_from_directory
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json


#img = cv2.imread("/Users/aliyahaas/Desktop/IMG_0124.jpeg")  # Replace with the path to your image
#cv2.imshow("IMG_0124.jpeg", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Load human face cascade file using cv2.CascadeClassifier built-in function
face_cascade_classifier = cv2.CascadeClassifier('/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/haarcascade_frontalface.xml')

# Load the face expression trained model
face_model = model_from_json(open("/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/facial_expression.json", "r").read())
face_model.load_weights('/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/facial_expression.h5')

# Define expressions
expressions = ('Angry:', 'Disgust:', 'Fear:', 'Happy:', 'Sad:', 'Surprise:', 'Neutral:')

# Load the video for facial expression recognition
video = cv2.VideoCapture('/Users/aliyahaas/Desktop/video_1.mp4')

frame = 0
detected_emotions = []

while True:
    ret, img = video.read()

    if not ret:
        break

    img = cv2.resize(img, (640, 360))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w > 130:
            face_detected = img[int(y):int(y + h), int(x):int(x + w)]
            face_detected = cv2.cvtColor(face_detected, cv2.COLOR_BGR2GRAY)
            face_detected = cv2.resize(face_detected, (48, 48))
            img_pixels = img_to_array(face_detected)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            predictions = face_model.predict(img_pixels)
            max_index = np.argmax(predictions[0])

            # Append the detected emotion to the list
            detected_emotions.append(expressions[max_index])

    frame += 1

    if frame > 227:
        break

video.release()
cv2.destroyAllWindows()

# After processing all frames, print the detected emotions
if detected_emotions:
    print("Detected Emotions:")
    for emotion in detected_emotions:
        print(emotion)
else:
    print("No emotions detected in the video.")
