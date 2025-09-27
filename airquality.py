import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Load the dataset
df = pd.read_csv('air_quality_health_impact_data.csv')
data_head = df.head()
missing_data = df.isnull().sum()
unique_values = df.nunique()

# Preprocess the data
df.fillna(df.mean(), inplace=True)
X = df.drop('HealthImpactClass', axis=1)
y = df['HealthImpactClass']


def run_logistic_regression(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return model, accuracy, report, conf_matrix


# Display data summary
def data_summary():
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)   
    print("Confusion Matrix:\n", conf_matrix)   


# Visualize the results
def heatmap(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
def vis_comparison():
    counts = df['HealthImpactClass'].value_counts().sort_index()
    plt.figure(figsize=(8,6))
    plt.bar(counts.index, counts.values, color='skyblue')
    plt.xlabel('Health Impact Class')
    plt.ylabel('Count')
    plt.title('Distribution of Health Impact Classes')
    plt.xticks(counts.index)
    plt.tight_layout()
    plt.show()

model, accuracy, report, conf_matrix = run_logistic_regression(X, y)
data_summary()


