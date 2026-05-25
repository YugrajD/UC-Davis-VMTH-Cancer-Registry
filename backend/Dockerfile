FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libgdal-dev \
    gosu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user. The entrypoint runs as root briefly to fix the
# volume-mounted /app/uploads ownership, then drops to `app` via gosu.
RUN addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /app/uploads && chown -R app:app /app \
    && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
# Shell form so ${WORKERS:-1} is expanded from the container environment at
# runtime.  Dev docker-compose overrides this CMD with uvicorn --reload.
# On Cloud Run set WORKERS to match the vCPU count (only when Redis is
# available for distributed rate-limiting/caching; default 1 is safe).
CMD gunicorn -w ${WORKERS:-1} -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    app.main:app
