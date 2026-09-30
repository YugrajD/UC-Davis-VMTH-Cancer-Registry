# Local HTTP worker. docker-compose mounts ./ml at /ml (the code app.py imports) and ./ml-worker at
# /worker; the bundle is found from PETBERT_MODEL_PATH.
FROM python:3.12-slim

WORKDIR /worker

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
