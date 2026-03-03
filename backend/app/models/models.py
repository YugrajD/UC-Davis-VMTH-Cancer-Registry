"""SQLAlchemy + GeoAlchemy2 models for the VMTH Cancer Registry."""

from sqlalchemy import (
    Boolean, Column, Integer, String, Numeric, Date, Text, ForeignKey, CheckConstraint, DateTime
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

from app.database import Base


class Species(Base):
    __tablename__ = "species"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)

    breeds = relationship("Breed", back_populates="species")
    patients = relationship("Patient", back_populates="species")


class Breed(Base):
    __tablename__ = "breeds"

    id = Column(Integer, primary_key=True)
    species_id = Column(Integer, ForeignKey("species.id"), nullable=False)
    name = Column(String(100), nullable=False)

    species = relationship("Species", back_populates="breeds")
    patients = relationship("Patient", back_populates="breed")


class CancerType(Base):
    __tablename__ = "cancer_types"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)

    cases = relationship("CancerCase", back_populates="cancer_type")
    case_diagnoses = relationship("CaseDiagnosis", back_populates="cancer_type")


class County(Base):
    __tablename__ = "counties"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    fips_code = Column(String(5), nullable=False, unique=True)
    geom = Column(Geometry("MULTIPOLYGON", srid=4326))
    population = Column(Integer)
    area_sq_miles = Column(Numeric(10, 2))
    is_catchment = Column(Boolean, nullable=False, server_default="false")

    patients = relationship("Patient", back_populates="county")
    cases = relationship("CancerCase", back_populates="county")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    species_id = Column(Integer, ForeignKey("species.id"), nullable=True)
    breed_id = Column(Integer, ForeignKey("breeds.id"), nullable=True)
    sex = Column(String(20), nullable=True)
    age_years = Column(Numeric(5, 1), nullable=True)
    weight_kg = Column(Numeric(6, 2))
    county_id = Column(Integer, ForeignKey("counties.id"), nullable=True)
    registered_date = Column(Date, nullable=True)
    anon_id = Column(String(100), nullable=True, unique=True, index=True)
    zip_code = Column(String(10), nullable=True)
    data_source = Column(String(20), nullable=True, default="mock")

    species = relationship("Species", back_populates="patients")
    breed = relationship("Breed", back_populates="patients")
    county = relationship("County", back_populates="patients")
    cases = relationship("CancerCase", back_populates="patient")


class CancerCase(Base):
    __tablename__ = "cancer_cases"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    cancer_type_id = Column(Integer, ForeignKey("cancer_types.id"), nullable=True)  # optional; types in case_diagnoses
    diagnosis_date = Column(Date, nullable=True)
    stage = Column(String(5))
    outcome = Column(String(20))
    county_id = Column(Integer, ForeignKey("counties.id"), nullable=True)
    source_row_index = Column(Integer, nullable=True)
    diagnosis_index = Column(Integer, nullable=True)
    icd_o_code = Column(String(20), nullable=True)
    predicted_term = Column(Text, nullable=True)
    original_text = Column(Text, nullable=True)
    confidence = Column(Numeric(4, 2), nullable=True)
    prediction_method = Column(String(20), nullable=True)

    patient = relationship("Patient", back_populates="cases")
    cancer_type = relationship("CancerType", back_populates="cases")
    county = relationship("County", back_populates="cases")
    reports = relationship("PathologyReport", back_populates="case")
    diagnoses = relationship("CaseDiagnosis", back_populates="case", cascade="all, delete-orphan")


class CaseDiagnosis(Base):
    """One row per cancer prediction (e.g. PetBERT) under a single registry case."""
    __tablename__ = "case_diagnoses"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cancer_cases.id", ondelete="CASCADE"), nullable=False)
    cancer_type_id = Column(Integer, ForeignKey("cancer_types.id"), nullable=False)
    icd_o_code = Column(String(20), nullable=True)
    predicted_term = Column(Text, nullable=True)
    original_text = Column(Text, nullable=True)
    confidence = Column(Numeric(4, 2), nullable=True)
    prediction_method = Column(String(20), nullable=True)
    source_row_index = Column(Integer, nullable=True)
    diagnosis_index = Column(Integer, nullable=True)

    case = relationship("CancerCase", back_populates="diagnoses")
    cancer_type = relationship("CancerType", back_populates="case_diagnoses")


class PathologyReport(Base):
    __tablename__ = "pathology_reports"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cancer_cases.id"), nullable=False)
    report_text = Column(Text, nullable=False)
    classification = Column(String(100))
    confidence_score = Column(Numeric(5, 4))
    report_date = Column(Date, nullable=False)

    case = relationship("CancerCase", back_populates="reports")


class CalEnviroScreen(Base):
    __tablename__ = "calenviroscreen"

    id = Column(Integer, primary_key=True)
    county_id = Column(Integer, ForeignKey("counties.id"), nullable=False, unique=True)
    ces_score = Column(Numeric(6, 2))
    pollution_burden = Column(Numeric(6, 2))
    ozone = Column(Numeric(6, 2))
    pm25 = Column(Numeric(6, 2))
    diesel_pm = Column(Numeric(6, 2))
    pesticides = Column(Numeric(6, 2))
    toxic_releases = Column(Numeric(6, 2))
    traffic = Column(Numeric(6, 2))
    drinking_water = Column(Numeric(6, 2))
    lead = Column(Numeric(6, 2))
    cleanup_sites = Column(Numeric(6, 2))
    groundwater_threats = Column(Numeric(6, 2))
    hazardous_waste = Column(Numeric(6, 2))
    solid_waste = Column(Numeric(6, 2))
    impaired_water = Column(Numeric(6, 2))
    pop_characteristics = Column(Numeric(6, 2))
    asthma = Column(Numeric(6, 2))
    low_birth_weight = Column(Numeric(6, 2))
    cardiovascular = Column(Numeric(6, 2))
    poverty = Column(Numeric(6, 2))
    unemployment = Column(Numeric(6, 2))
    housing_burden = Column(Numeric(6, 2))
    education = Column(Numeric(6, 2))
    linguistic_isolation = Column(Numeric(6, 2))

    county = relationship("County")


class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True)
    dataset_a_filename = Column(String(255))
    dataset_b_filename = Column(String(255))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    rows_processed = Column(Integer, default=0)
    rows_inserted = Column(Integer, default=0)
    rows_skipped = Column(Integer, default=0)
    rows_errored = Column(Integer, default=0)
    errors = Column(JSONB, default=list)
    warnings = Column(JSONB, default=list)
