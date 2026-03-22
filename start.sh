#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
  echo "Error: .env file not found. Run 'cp .env.example .env' and fill in your Supabase credentials."
  exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running. Start Docker Desktop and try again."
  exit 1
fi

echo "Starting backend, frontend, and ML worker..."
docker compose up --build backend frontend ml-worker
