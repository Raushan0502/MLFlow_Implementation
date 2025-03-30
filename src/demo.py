import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns
import os

# if not mlflow.is_tracking_uri_set():
#     mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI here   
#Integrate mlflow with remote server dagshub
import dagshub
dagshub.init(repo_owner='rk7250846', repo_name='MLFlow_Implementation', mlflow=True)
# Set the tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/rk7250846/MLFlow_Implementation.mlflow")


dataset = load_wine()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 


max_depth = 5
n_estimators = 50

mlflow.set_experiment("Wine Quality Prediction-1")  # Set the experiment name

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(x_train, y_train)        
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    plt.title('Confusion Matrix')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted') 
    plt.ylabel('True')

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)  
    mlflow.log_metric("accuracy", accuracy)

    #log confusion matrix as artifact
    plt.savefig("confusion_matrix.png") 
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(rf, "model")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    #Add tag to the run
    mlflow.set_tag("model_type", "RandomForestClassifier")  
    mlflow.set_tag("dataset", "Wine Quality")
    mlflow.set_tag("author", "Raushan Kumar")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)    
