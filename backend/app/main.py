"""FastAPI application entry point for the VMTH Cancer Registry."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import dashboard, incidence, geo, trends, search

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="UC Davis Veterinary Medical Teaching Hospital Cancer Registry API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(incidence.router)
app.include_router(geo.router)
app.include_router(trends.router)
app.include_router(search.router)


@app.get("/")
async def root():
    return {
        "name": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
