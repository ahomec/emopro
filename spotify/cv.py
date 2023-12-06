from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import pandas as pd
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution
tf.compat.v1.disable_v2_behavior

# Load data
df = pd.DataFrame(pd.read_csv("~/OneDrive - Creighton University/Year 4/2023 Fall/DSC 599/emopro/spotify/spotify_data.csv"))

# Set target variable
col_features = df.columns[6:-3]
x = df[col_features]
y = df["target"]

x = MinMaxScaler().fit_transform(x)

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, encoded_y, test_size = 0.3, random_state = 12)
data = list(zip(x, y))

gen_X = np.array(list(range(1,101)))
gen_Y = np.array(list(range(1,101)))

# Gaussian kernel width tuning

results = [] # we will store our results here.

for k in range(1, 12): # maximum neighbor is 9 since 9:1 train split
    
    # you can create your own list
    # for w in np.array([0.1, 0.5, 1, 10, 100])
    for w in np.random.randint(5, 100000, 100)/1000:
        
        temp_mae = [] # temporary storage for mean absolute error
        
        # Set kernel_width = w
        def gaussian_kernel(distances):
            kernel_width = w
            weights = np.exp(-(distances**2)/kernel_width)
            return weights
        
        # We take average of ten bootstrapped model.
        for i in range(0,100):
            # Below is my personal way of setting random state. This makes each run a bootstrapped model.
            X_train, X_test, y_train, y_test = train_test_split(gen_X.reshape(-1,1),
                                                                gen_Y,
                                                                test_size=0.1,
                                                                random_state=int(100*np.random.random()))
            knn = KNeighborsRegressor(n_neighbors=k,weights=gaussian_kernel)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            mean_absolute_error = np.mean(abs(y_pred-y_test))
            temp_mae.append(mean_absolute_error)
        average_over_ten_runs = np.mean(temp_mae)
        results.append([k, w, average_over_ten_runs]) # We store our run result.
results = np.array(results)
results[:,2].min()
results[results[:,2].argmin()]

# Optimal width = array([2.        , 7.576     , 0.13824287])