#%%
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import pandas as pd
import numpy as np

# Load data
df_test = pd.DataFrame(pd.read_csv("~/OneDrive - Creighton University/Year 4/2023 Fall/DSC 599/emopro/spotify/spotify_data.csv"))
df_moods = pd.DataFrame(pd.read_csv("~/OneDrive - Creighton University/Year 4/2023 Fall/DSC 599/emopro/spotify/data_moods.csv"))

# Set target variable
col_features = df_moods.columns[6:-3]
x = df_moods[col_features]
y = df_moods['mood']

# Normalize data
x = MinMaxScaler().fit_transform(x)

# Encode non-numeric data
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# Define gaussian kernel
# from https://github.com/seho0808/knn_gaussian_medium/blob/master/Medium_KNN.ipynb
def gaussian_kernel(distances):
    kernel_width = 2.926 # Tuned in cross-validation.py
    weights = np.exp(-(distances**2)/kernel_width)
    return weights

# Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, encoded_y, test_size = 0.3, random_state = 12)
data = list(zip(x, y))

# k-NN with Default weights
knn_def = KNeighborsClassifier(n_neighbors = 19)
knn_def.fit(x_train, y_train)
print(knn_def.score(x_test, y_test)) 

# k-NN with Inverse weights
knn_inv = KNeighborsClassifier(n_neighbors = 24, weights = 'distance')
knn_inv.fit(x_train, y_train)
print(knn_inv.score(x_test, y_test)) 

# k-NN Gaussian weights
knn_gk = KNeighborsClassifier(n_neighbors = 19, weights = gaussian_kernel)
knn_gk.fit(x_train, y_train)
print(knn_gk.score(x_test,y_test))

# Choosing optimal k value
# From https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn

# Set k-value range
k_values = [i for i in range (1,31)]

# Default weights
scores = []
scaler = StandardScaler()
X = scaler.fit_transform(x)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))


import seaborn as sns
from matplotlib import pyplot as plt
sns.lineplot(x = k_values, y = scores, marker = 'o')

# Inverse weights
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(x)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance')
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))

sns.lineplot(x = k_values, y = scores, marker = 'o')

# Gaussian weights
k_values = [i for i in range (1,31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(x)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights = gaussian_kernel)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))


# Visualize accuracy vs. k-values
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.legend(labels=["default", "inverse", "Gaussian"])
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
# %%
