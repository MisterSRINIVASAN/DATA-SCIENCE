import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data import get_train_test_split

# Set up MLflow tracking URI to a local directory
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

def evaluate_metrics(y_true, y_pred):
    """
    Calculate and return multiple classification metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')
    return accuracy, precision, recall, f1

def train(n_estimators: int, max_depth: int, random_state: int = 42):
    """
    Train a Random Forest model on the synthetic TN Election dataset and log to MLflow.
    """
    # 1. Get data
    print("Generating dataset and splitting...")
    X_train, X_test, y_train, y_test = get_train_test_split(random_seed=random_state)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # 2. Set experiment
    mlflow.set_experiment("TN_Election_Prediction")
    
    # 3. Start MLflow Run
    with mlflow.start_run():
        print(f"Training RandomForest with n_estimators={n_estimators}, max_depth={max_depth}...")
        
        # 4. Initialize and Train Model
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # 5. Predict and Evaluate
        y_pred = rf.predict(X_test)
        accuracy, precision, recall, f1 = evaluate_metrics(y_test, y_pred)
        
        print(f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # 6. Log Parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # 7. Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # 8. Log Model
        # We specify the model flavor and the name it will be stored under
        mlflow.sklearn.log_model(rf, "rf_model")
        
        print("Model and metrics successfully logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
