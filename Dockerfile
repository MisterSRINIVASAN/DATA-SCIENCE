FROM python:3.10-slim

# Set working directory
WORKDIR /mlops-pipeline

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ ./app/

# Copy MLflow tracking directory (needed to load the model)
# Make sure to run `python src/train.py` before building the docker image!
COPY mlruns/ ./mlruns/

# Set environment variables
ENV PYTHONPATH=/mlops-pipeline

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
