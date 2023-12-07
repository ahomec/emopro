from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import pandas as pd
import numpy as np

# Load data
df = pd.DataFrame(pd.read_csv("~/OneDrive - Creighton University/Year 4/2023 Fall/DSC 599/emopro/spotify/spotify_data.csv"))

# Set target variable
col_features = df.columns[6:-3]
x = df[col_features]
y = df["target"]

# Normalize data
x = MinMaxScaler().fit_transform(x)

# Encode non-numeric data
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, encoded_y, test_size = 0.3, random_state = 12)
data = list(zip(x, y))

# Gaussian kernel width tuning
# From https://github.com/seho0808/knn_gaussian_medium/blob/master/Medium_KNN.ipynb
gen_X = np.array(list(range(1,101)))
gen_Y = np.array(list(range(1,101)))

results = [] # we will store our results here.

for k in range(1, 12): 
    # for w in np.array([0.1, 0.5, 1, 10, 100])
    for w in np.random.randint(5, 100000, 100)/1000:
        
        temp_mae = [] # temporary storage for mean absolute error
        
        # Set kernel_width = w
        def gaussian_kernel(distances):
            kernel_width = w
            weights = np.exp(-(distances**2)/kernel_width)
            return weights
        
        # Bootstrap
        for i in range(0,100):
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
        results.append([k, w, average_over_ten_runs])
        
# Print resulting arrays
results = np.array(results)
results[:,2].min()
results[results[:,2].argmin()]