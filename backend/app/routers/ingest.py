"""Router for data ingestion — placeholder for future frontend upload.

Primary ingestion is done via the standalone script:
    docker compose run --rm ingest
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


@router.get("/status")
async def ingestion_status():
    """Health check for the ingestion module."""
    return {
        "status": "ok",
        "message": "Use 'docker compose run --rm ingest' to ingest PetBERT data.",
    }
