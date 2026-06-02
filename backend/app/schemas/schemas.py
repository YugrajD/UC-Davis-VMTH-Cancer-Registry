"""Pydantic request/response models for the API."""

from typing import Literal, Optional, List, Any
from datetime import date

from pydantic import BaseModel, Field


# --- Lookup Schemas ---

class SpeciesOut(BaseModel):
    id: int
    name: str
    model_config = {"from_attributes": True}


class BreedOut(BaseModel):
    id: int
    species_id: int
    name: str
    model_config = {"from_attributes": True}


class CancerTypeOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    model_config = {"from_attributes": True}


class CountyOut(BaseModel):
    id: int
    name: str
    fips_code: str
    population: Optional[int] = None
    area_sq_miles: Optional[float] = None
    model_config = {"from_attributes": True}


# --- Dashboard Summary ---

class SpeciesBreakdown(BaseModel):
    species: str
    count: int
    percentage: float


class TopCancer(BaseModel):
    cancer_type: str
    count: int


class DashboardSummary(BaseModel):
    total_cases: int
    total_patients: int
    total_counties: int
    year_range: List[int]
    species_breakdown: List[SpeciesBreakdown]
    top_cancers: List[TopCancer]
    top_county: str
    top_county_cases: int


# --- Incidence ---

class IncidenceRecord(BaseModel):
    cancer_type: str
    county: Optional[str] = None
    zip_code: Optional[str] = None
    species: Optional[str] = None
    breed: Optional[str] = None
    year: Optional[int] = None
    count: int


class IncidenceResponse(BaseModel):
    data: List[IncidenceRecord]
    total: int
    filters_applied: dict


# --- GeoJSON ---

class GeoJSONFeatureProperties(BaseModel):
    name: str
    fips_code: str
    total_cases: int
    top_cancer: Optional[str] = None


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: Any
    properties: GeoJSONFeatureProperties


class GeoJSONResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]


class CalEnviroScreenOut(BaseModel):
    county_id: int
    county_name: str
    county_fips: str
    ces_score: Optional[float] = None
    pollution_burden: Optional[float] = None
    ozone: Optional[float] = None
    pm25: Optional[float] = None
    diesel_pm: Optional[float] = None
    pesticides: Optional[float] = None
    toxic_releases: Optional[float] = None
    traffic: Optional[float] = None
    drinking_water: Optional[float] = None
    lead: Optional[float] = None
    cleanup_sites: Optional[float] = None
    groundwater_threats: Optional[float] = None
    hazardous_waste: Optional[float] = None
    solid_waste: Optional[float] = None
    impaired_water: Optional[float] = None
    pop_characteristics: Optional[float] = None
    asthma: Optional[float] = None
    low_birth_weight: Optional[float] = None
    cardiovascular: Optional[float] = None
    poverty: Optional[float] = None
    unemployment: Optional[float] = None
    housing_burden: Optional[float] = None
    education: Optional[float] = None
    linguistic_isolation: Optional[float] = None
    model_config = {"from_attributes": True}


class CountyDetail(BaseModel):
    county: CountyOut
    total_cases: int
    cancer_breakdown: List[TopCancer]
    species_breakdown: List[SpeciesBreakdown]
    yearly_trend: List[dict]


# --- Trends ---

class TrendPoint(BaseModel):
    year: int
    count: int
    deceased: Optional[int] = None
    alive: Optional[int] = None


class TrendSeries(BaseModel):
    name: str
    data: List[TrendPoint]


class TrendsResponse(BaseModel):
    series: List[TrendSeries]


# --- Search / BERT ---

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


class ClassifyResult(BaseModel):
    predicted_cancer_type: str
    confidence: float
    top_predictions: List[dict]


# --- Breed Detail ---

class BreedCancerTypeCount(BaseModel):
    cancer_type: str
    count: int

class BreedCountyCount(BaseModel):
    county_name: str
    fips_code: str
    count: int

class BreedSexCount(BaseModel):
    sex: str
    count: int

class BreedDetailOut(BaseModel):
    breed: str
    total_cases: int
    sex_breakdown: List[BreedSexCount]
    cancer_types: List[BreedCancerTypeCount]
    county_cases: List[BreedCountyCount]


# --- Filter Options ---

class FilterOptions(BaseModel):
    species: List[SpeciesOut]
    cancer_types: List[CancerTypeOut]
    counties: List[CountyOut]
    breeds: List[BreedOut]
    year_range: List[int]


# --- Ingestion ---

class IngestionRowResult(BaseModel):
    row_number: int
    anon_id: str
    status: str  # "inserted", "skipped", "error"
    message: Optional[str] = None
    cancer_type: Optional[str] = None
    confidence: Optional[float] = None


class IngestionResponse(BaseModel):
    total_rows: int
    inserted: int
    skipped: int
    errors: int
    warnings: List[str] = []
    row_results: List[IngestionRowResult] = []
    ingestion_log_id: Optional[int] = None
    result_summary: Optional[dict] = None


# --- Ingestion Jobs ---

class IngestionJobOut(BaseModel):
    id: int
    uploaded_by_email: str
    dataset_a_filename: str
    status: str
    reviewed_by_email: Optional[str] = None
    reviewed_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    ingestion_log_id: Optional[int] = None
    processing_error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    model_config = {"from_attributes": True}


class IngestionJobReview(BaseModel):
    action: Literal["approve", "reject"]
    rejection_reason: Optional[str] = Field(default=None, max_length=2000)
    model_folder: Optional[str] = Field(default=None, max_length=255)
    clinic_name: Optional[str] = Field(default=None, max_length=255)
