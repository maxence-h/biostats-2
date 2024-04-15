import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 1000

factor_a = np.random.normal(loc=0, scale=1, size=n_samples)
factor_b = np.random.normal(loc=5, scale=2, size=n_samples)
factor_c = np.random.normal(loc=-5, scale=3, size=n_samples)

cancer_subtype = np.random.choice(['Type A', 'Type B', 'Type C'], size=n_samples, p=[0.05, 0.65, 0.3])

df = pd.DataFrame({
    'Factor A': factor_a,
    'Factor B': factor_b,
    'Factor C': factor_c,
    'Cancer Subtype': cancer_subtype
})

scaler = StandardScaler()
features_scaled = scaler.fit_transform(df.iloc[:, :-1])

kpca = KernelPCA(n_components=2, kernel='rbf')
kpca_features = kpca.fit_transform(features_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(kpca_features)

plt.scatter(kpca_features[:, 0], kpca_features[:, 1], c=clusters)
plt.title('Clusters by KPCA Features')
plt.xlabel('KPCA Component 1')
plt.ylabel('KPCA Component 2')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(kpca_features, clusters, test_size=0.3, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)
