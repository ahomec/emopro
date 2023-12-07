import numpy as np
import cv2
from keras.preprocessing import image_dataset_from_directory
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load human face cascade file using cv2.CascadeClassifier built-in function
face_cascade_classifier = cv2.CascadeClassifier('/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/haarcascade_frontalface.xml')

# Load the face expression trained model
face_model = model_from_json(open("/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/facial_expression.json", "r").read())
face_model.load_weights('/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/facial_expression.h5')

# Define expressions
expressions = ('Angry:', 'Disgust:', 'Fear:', 'Happy:', 'Sad:', 'Surprise:', 'Neutral:')

# Load the video for facial expression recognition
video = cv2.VideoCapture('/Users/aliyahaas/Desktop/video_1.mp4')

# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        return lr * 0.9
    else:
        return lr

# Assuming 'face_model' is your existing model
face_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize true_labels
true_labels = []

# Initialize detected_emotions
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
            detected_emotions.append(max_index)

            # Replace 0 with the actual true label for this frame
            true_labels.append(3)

# After processing all frames, compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, detected_emotions, labels=list(range(len(expressions))))
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix with seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=expressions, yticklabels=expressions)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# You can also print other metrics such as precision, recall, and F1-score
classification_rep = classification_report(true_labels, detected_emotions, labels=list(range(len(expressions))), target_names=expressions)
print("Classification Report:")
print(classification_rep)
