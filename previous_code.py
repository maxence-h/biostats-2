import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# Load the dataset
data_path = 'C:/Users/maxmo/PycharmProjects/biostats-2/data_biostats.csv'
data = pd.read_csv(data_path)

# Assume the target variable is in the last column and the features are all other columns
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that includes scaling, feature selection, and a classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Data normalization
    ('selector', SelectKBest(score_func=f_classif, k='all')),  # Feature selection
    ('classifier', RandomForestClassifier(random_state=42))  # Model training
])

# Define a parameter grid to search for the best parameters for feature selection and the classifier
param_grid = {
    'selector__k': [10, 50, 100, 'all'],  # Number of features to select
    'classifier__n_estimators': [50, 100, 200],  # Number of trees in the random forest
    'classifier__max_depth': [None, 10, 20, 30]  # Maximum depth of the trees
}

# Set up the grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Predict on the test data using the best model
y_pred = grid_search.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
