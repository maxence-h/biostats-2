from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
import numpy as np

def build_dae(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    # DAE Model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# Specify dimensions
input_dim = X_multi_omics.shape[1]  # Number of features in your input data
encoding_dim = 100  # Desired dimension of encoded representation

# Build and train DAE
dae = build_dae(input_dim, encoding_dim)
dae.fit(X_multi_omics, X_multi_omics, epochs=50, batch_size=256, shuffle=True, validation_data=(X_multi_omics, X_multi_omics))

# Encode multi-omics data to reduced dimensionality
encoder = Model(dae.input, dae.layers[-2].output)
X_encoded = encoder.predict(X_multi_omics)

# Step 2: K-means Clustering
kmeans = KMeans(n_clusters=3)  # Specify the number of clusters based on silhouette score or other criteria
cluster_labels = kmeans.fit_predict(X_encoded)

# Step 3: Logistic Regression for Subtype Classification
X_train, X_test, y_train, y_test = train_test_split(X_mRNA, cluster_labels, test_size=0.2, random_state=42)
log_reg = LogisticRegression(penalty='l1', solver='liblinear')  # Using L1 regularization
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

