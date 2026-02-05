"""Pydantic request/response models for the API."""

from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import date


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
    population: Optional[int] = None
    total_cases: int
    cases_per_capita: Optional[float] = None
    top_cancer: Optional[str] = None


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: Any
    properties: GeoJSONFeatureProperties


class GeoJSONResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]


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
    text: str


class ClassifyResult(BaseModel):
    predicted_cancer_type: str
    confidence: float
    top_predictions: List[dict]


class ReportOut(BaseModel):
    id: int
    case_id: int
    report_text: str
    classification: Optional[str] = None
    confidence_score: Optional[float] = None
    report_date: date
    model_config = {"from_attributes": True}


class ReportSearchResponse(BaseModel):
    reports: List[ReportOut]
    total: int


# --- Filter Options ---

class FilterOptions(BaseModel):
    species: List[SpeciesOut]
    cancer_types: List[CancerTypeOut]
    counties: List[CountyOut]
    breeds: List[BreedOut]
    year_range: List[int]
