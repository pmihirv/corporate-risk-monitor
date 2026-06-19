FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000

# Retrain the model on container startup to guarantee version consistency, then launch the app
CMD ["sh", "-c", "python train_pipeline.py && uvicorn app:app --host 0.0.0.0 --port 8000"]
