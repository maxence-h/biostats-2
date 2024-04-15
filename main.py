import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_path = 'C:/Users/maxmo/PycharmProjects/biostats-2/data_biostats.csv'
data = pd.read_csv(data_path)

data = data.select_dtypes(include=[float, int])


X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_scaled)

X_train_with_cluster = pd.concat([pd.DataFrame(X_train_scaled), pd.Series(kmeans.labels_, name='cluster')], axis=1)
X_test_with_cluster = pd.concat([pd.DataFrame(X_test_scaled), pd.Series(kmeans.predict(X_test_scaled), name='cluster')], axis=1)

logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_with_cluster, y_train)

y_pred = logreg.predict(X_test_with_cluster)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
