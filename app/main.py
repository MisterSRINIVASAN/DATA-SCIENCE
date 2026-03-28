import os
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.schemas import ElectionPredictionRequest, ElectionPredictionResponse

app = FastAPI(title="TN Election Prediction API", version="1.0.0")

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    # We assume mlruns is in the current working directory where the app is launched
    os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
    
    try:
        experiment = mlflow.get_experiment_by_name("TN_Election_Prediction")
        if not experiment:
            print("WARNING: Experiment 'TN_Election_Prediction' not found. Model not loaded.")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["metrics.accuracy DESC"]
        )
        
        if runs.empty:
            print("WARNING: No runs found. Model not loaded.")
            return

        best_run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{best_run_id}/rf_model"
        
        print(f"Loading best model from {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")

# Mount static directory for frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "TN Election Prediction API is running."}

@app.post("/predict", response_model=ElectionPredictionResponse)
def predict(request: ElectionPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please ensure it was trained and logged via MLflow.")
        
    # Convert Pydantic object to pandas DataFrame (Pydantic v2 syntax)
    data = pd.DataFrame([request.model_dump()])
    
    # Predict using the loaded MLflow model
    prediction_val = int(model.predict(data)[0])
    
    # Get probabilities
    probabilities = model.predict_proba(data)[0]
    prob_dmk = float(probabilities[0])
    prob_admk = float(probabilities[1])
    prob_tvk = float(probabilities[2])
    prob_ntk = float(probabilities[3])
    prob_pmk = float(probabilities[4])
    prob_bjp = float(probabilities[5])
    prob_inc = float(probabilities[6])
    
    # Label mapping (0 -> DMK, 1 -> ADMK, 2 -> TVK, 3 -> NTK, 4 -> PMK, 5 -> BJP, 6 -> INC)
    labels_map = {0: "DMK", 1: "ADMK", 2: "TVK", 3: "NTK", 4: "PMK", 5: "BJP", 6: "INC"}
    label = labels_map.get(prediction_val, "Unknown")
    
    return ElectionPredictionResponse(
        prediction=prediction_val,
        prediction_label=label,
        probability_dmk=prob_dmk,
        probability_admk=prob_admk,
        probability_tvk=prob_tvk,
        probability_ntk=prob_ntk,
        probability_pmk=prob_pmk,
        probability_bjp=prob_bjp,
        probability_inc=prob_inc
    )
