# End-to-End MLOps Pipeline

This project demonstrates a complete machine learning lifecycle using a synthetic **Tamil Nadu Assembly Elections** dataset.

## Features
- **Scikit-Learn**: Custom dataset generation and Random Forest model training.
- **MLflow**: Experiment tracking (hyperparameters, metrics, and models).
- **FastAPI**: REST API to serve the model.
- **Beautiful Web Dashboard**: A modern glassmorphism frontend using HTML/CSS and Chart.js to visualize predictions dynamically.
- **Docker**: Containerization of the entire application.
- **Power BI / Tableau Ready**: Easily export the synthetic dataset to CSV for building beautiful BI dashboards.

## Quickstart

### 1. Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Export Data for Power BI / Tableau (Optional)
If you want to create beautiful dashboards in Power BI or Tableau, run this script to export the dataset to a CSV file (`tn_election_data.csv`).
```bash
python src/export_data.py
```

### 3. Model Training & MLflow Tracking
Train the Random Forest model. This will automatically track metrics and save the model using MLflow.
```bash
python src/train.py --n_estimators 150 --max_depth 10
```

You can view the MLflow UI by running:
```bash
mlflow ui
```
Navigate to `http://127.0.0.1:5000` to see your experiments.

### 4. Run the API and Web Dashboard Locally
Once the model is trained, start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
Navigate to `http://127.0.0.1:8000/` in your browser. You will see a beautiful, interactive dashboard where you can input constituency parameters and see real-time predictions visualized with Chart.js.

### 5. Docker Containerization
To package everything into a production-ready Docker container (ensure you have trained the model first, as Docker needs the `mlruns` directory to copy the model):
```bash
docker build -t tn-mlops-pipeline .
docker run -p 8000:8000 tn-mlops-pipeline
```
Then navigate to `http://localhost:8000/` to use the dashboard!
