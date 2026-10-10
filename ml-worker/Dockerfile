# Local HTTP worker. docker-compose mounts ./ml at /ml (the code app.py imports) and ./ml-worker at
# /worker; the bundle (a report-mapping generation directory) is found from PETBERT_MODEL_PATH, whose
# parent directory it is. CPU torch, fp32; threads from TORCH_NUM_THREADS (default min(8, CPU count)).
FROM python:3.12-slim

WORKDIR /worker

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
