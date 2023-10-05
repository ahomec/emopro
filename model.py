from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# from keras.utils import np_utils

import tensorflow as tf
tf.compat.v1.disable_eager_execution
tf.compat.v1.disable_v2_behavior

df = pd.DataFrame(pd.read_csv("spotify_data.csv"))

col_features = df.columns[6:-3]
x = df[col_features]
y = df["target"]

x = MinMaxScaler().fit_transform(x)

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, encoded_y, test_size = 0.3, random_state = 15)

