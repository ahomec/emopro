import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load human face cascade file using cv2.CascadeClassifier built-in function
face_cascade_classifier = cv2.CascadeClassifier('/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/haarcascade_frontalface.xml')

# Load the face expression trained model
face_model = model_from_json(open("/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/facial_expression.json", "r").read())
face_model.load_weights('/Users/aliyahaas/Desktop/Human_Facial_Emotion_Recognition/facial_expression.h5')

# Define expressions
expressions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

# Mapping for emotions
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

#  ImageDataGenerator with data augmentation 
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

frame = 0
detected_emotions = []
true_labels = []

# Assuming 'face_model' is your existing model
face_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

while True:
    ret, img = video.read()

    if not ret:
        break

    img = cv2.resize(img, (640, 360))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w > 130:
             #region of interest
            face_detected = img[int(y):int(y + h), int(x):int(x + w)]
            
            # Apply data augmentation with random transform 
            face_detected = datagen.random_transform(face_detected)
            #color to grayscale
            face_detected = cv2.cvtColor(face_detected, cv2.COLOR_BGR2GRAY)
            #resize to 48x48 pixels
            face_detected = cv2.resize(face_detected, (48, 48))
             #input form 
            img_pixels = img_to_array(face_detected)
             #adds dimension to array
            img_pixels = np.expand_dims(img_pixels, axis=0)
             #normalization
            img_pixels /= 255
             #uses trained facial expression recognition model to predict the emotion
            predictions = face_model.predict(img_pixels)
            #index referring to the emotion with the highest probability
            max_index = np.argmax(predictions[0])

            # Append the detected emotion to the list
            detected_emotions.append(emotion_mapping[expressions[max_index]])

            # Append the true label corresponding to the detected emotion
            true_labels.append(emotion_mapping[expressions[max_index]])

#jeeos track of current frame number 
    frame += 1

# if the frame is greater the 227, it breaks
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
