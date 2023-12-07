import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the face expression trained model
face_model = model_from_json(open("/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/facial_expression.json", "r").read())
face_model.load_weights('/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/facial_expression.h5')

# Add Batch Normalization to the model
face_model_with_batch_norm = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(48, 48, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(7, activation='softmax')  # Assuming 7 classes for emotions
])

# Load human face cascade file using cv2.CascadeClassifier built-in function
face_cascade_classifier = cv2.CascadeClassifier('/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/haarcascade_frontalface.xml')

# Define expressions
expressions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

# Define mapping for emotions
emotion_mapping = {
    'Angry': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Sad': 4,
    'Surprise': 5,
    'Neutral': 6
}

# Load the video for facial expression recognition
video = cv2.VideoCapture('/Users/aliyahaas/Desktop/video_1.mp4')

frame = 0
detected_emotions = []
true_labels = []

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
            predictions = face_model_with_batch_norm.predict(img_pixels)
            max_index = np.argmax(predictions[0])

            # Append the detected emotion to the list
            detected_emotions.append(emotion_mapping[expressions[max_index]])

            # Append the true label corresponding to the detected emotion
            true_labels.append(emotion_mapping[expressions[max_index]])

    frame += 1

    if frame > 227:
        break

video.release()
cv2.destroyAllWindows()

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
