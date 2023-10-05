import numpy as py
import cv2
from keras.preprocessing import image_dataset_from_directory
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.model import model_from_json

img = cv2.imread("path_to_your_image.jpg")  # Replace with the path to your image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()