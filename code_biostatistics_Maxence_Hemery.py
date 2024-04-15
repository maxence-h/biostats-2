import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd

data_biostats = pd.read_csv('C:/Users/maxmo/PycharmProjects/biostats-2/gene_expression_data.csv')
data_biostats = data_biostats.select_dtypes(include=[np.number]) 
data_biostats = data_biostats.drop(columns=[col for col in data_biostats if 'Unnamed' in col])

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_biostats)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_pca)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=clusters, palette='viridis', style=clusters, markers=['o', 's', 'D'])
plt.title('PCA of Gene Expression Profiles and K-means Clustering')
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.legend(title='Cluster Predicted', labels=['Luminal A', 'Luminal B', 'HER2-enriched'])
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x=clusters)
plt.title("Distribution of Samples Across Identified Clusters by K-means")
plt.xlabel("Cluster")
plt.ylabel("Number of Samples")
plt.ylim(160,170)
plt.xticks([0, 1, 2], ['Luminal A', 'Luminal B', 'HER2-enriched'])
plt.show()

data_biostats = pd.read_csv('C:/Users/maxmo/PycharmProjects/biostats-2/data_biostats.csv')
data_biostats = data_biostats.select_dtypes(include=[np.number])
data_biostats = data_biostats.drop(columns=[col for col in data_biostats if 'Unnamed' in col])

variances = np.var(data_biostats, axis=0)
top_var_indices = np.argsort(variances)[-10:]
top_var_data = data_biostats.iloc[:, top_var_indices]

plt.figure(figsize=(10, 8))
sns.heatmap(top_var_data, cmap='viridis')
plt.title('Heatmap of Top 10 Variable Genes')
plt.xlabel('Genes')
plt.ylabel('Samples')
plt.show()
