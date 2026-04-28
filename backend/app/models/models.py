"""SQLAlchemy + GeoAlchemy2 models for the VMTH Cancer Registry."""

from sqlalchemy import (
    Boolean, Column, Integer, String, Numeric, Date, Text, ForeignKey, CheckConstraint, DateTime, func
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
    # FALSE for types auto-created from a reviewer correction; admin must
    # confirm before they appear in dashboard filters.
    confirmed = Column(Boolean, nullable=False, server_default="true")

    case_diagnoses = relationship(
        "CaseDiagnosis",
        back_populates="cancer_type",
        foreign_keys="CaseDiagnosis.cancer_type_id",
    )


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


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    species_id = Column(Integer, ForeignKey("species.id"), nullable=True)
    breed_id = Column(Integer, ForeignKey("breeds.id"), nullable=True)
    sex = Column(String(20), nullable=True)
    county_id = Column(Integer, ForeignKey("counties.id"), nullable=True)
    anon_id = Column(String(100), nullable=True, unique=True, index=True)
    zip_code = Column(String(10), nullable=True)
    data_source = Column(String(20), nullable=True, default="mock")
    diagnosis_date = Column(Date, nullable=True)
    outcome = Column(String(20), nullable=True)

    species = relationship("Species", back_populates="patients")
    breed = relationship("Breed", back_populates="patients")
    county = relationship("County", back_populates="patients")
    diagnoses = relationship("CaseDiagnosis", back_populates="patient")
    reports = relationship("PathologyReport", back_populates="patient")


class CaseDiagnosis(Base):
    """One row per cancer prediction (e.g. PetBERT) under a single patient."""
    __tablename__ = "case_diagnoses"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    cancer_type_id = Column(Integer, ForeignKey("cancer_types.id"), nullable=False)
    icd_o_code = Column(String(20), nullable=True)
    predicted_term = Column(Text, nullable=True)
    confidence = Column(Numeric(4, 2), nullable=True)
    prediction_method = Column(String(20), nullable=True)
    source_row_index = Column(Integer, nullable=True)
    diagnosis_index = Column(Integer, nullable=True)

    # Review workflow — see database/migrations/010_diagnosis_review.sql
    review_status = Column(String(20), nullable=False, server_default="confirmed")
    reviewed_by_email = Column(String(255), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    original_cancer_type_id = Column(Integer, ForeignKey("cancer_types.id"), nullable=True)
    original_icd_o_code = Column(String(20), nullable=True)
    original_predicted_term = Column(Text, nullable=True)
    top2_margin = Column(Numeric(4, 2), nullable=True)
    ingestion_job_id = Column(Integer, ForeignKey("ingestion_jobs.id"), nullable=True)

    patient = relationship("Patient", back_populates="diagnoses")
    ingestion_job = relationship("IngestionJob")
    cancer_type = relationship(
        "CancerType",
        back_populates="case_diagnoses",
        foreign_keys=[cancer_type_id],
    )
    review_events = relationship(
        "DiagnosisReviewEvent",
        back_populates="case_diagnosis",
        cascade="all, delete-orphan",
        order_by="DiagnosisReviewEvent.created_at.desc()",
    )

    __table_args__ = (
        CheckConstraint(
            "review_status IN ('pending', 'confirmed', 'corrected', 'rejected')",
            name="case_diagnoses_review_status_check",
        ),
    )


class DiagnosisReviewEvent(Base):
    """Append-only audit log of every state change to a case_diagnosis review."""
    __tablename__ = "diagnosis_review_events"

    id = Column(Integer, primary_key=True)
    case_diagnosis_id = Column(
        Integer,
        ForeignKey("case_diagnoses.id", ondelete="CASCADE"),
        nullable=False,
    )
    actor_email = Column(String(255), nullable=False)
    action = Column(String(20), nullable=False)
    from_status = Column(String(20), nullable=True)
    to_status = Column(String(20), nullable=False)
    cancer_type_id_before = Column(Integer, ForeignKey("cancer_types.id"), nullable=True)
    cancer_type_id_after = Column(Integer, ForeignKey("cancer_types.id"), nullable=True)
    icd_o_code_before = Column(String(20), nullable=True)
    icd_o_code_after = Column(String(20), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    case_diagnosis = relationship("CaseDiagnosis", back_populates="review_events")


class PathologyReport(Base):
    __tablename__ = "pathology_reports"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    report_text = Column(Text, nullable=False)
    classification = Column(String(100))
    confidence_score = Column(Numeric(5, 4))
    report_date = Column(Date, nullable=False)

    patient = relationship("Patient", back_populates="reports")


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


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(Integer, primary_key=True)
    uploaded_by_email = Column(String(255), nullable=False)
    uploaded_by_sub = Column(String(255), nullable=False)
    dataset_a_filename = Column(String(255), nullable=False)
    dataset_b_filename = Column(String(255), nullable=True)
    storage_path = Column(String(500), nullable=False)
    status = Column(String(20), nullable=False, default="pending_review")
    reviewed_by_email = Column(String(255))
    reviewed_at = Column(DateTime(timezone=True))
    rejection_reason = Column(Text)
    ingestion_log_id = Column(Integer, ForeignKey("ingestion_logs.id"))
    processing_error = Column(Text)
    batch_job_name = Column(String(500), nullable=True)
    processing_stage = Column(String(50), nullable=True)
    result_summary = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))


class UserRole(Base):
    """Per-email role assignments managed via the admin panel.

    DB rows take precedence over the *_EMAILS env vars, which are now
    only used as a startup seed.
    """
    __tablename__ = "user_roles"

    email = Column(String(255), primary_key=True)
    is_admin = Column(Boolean, nullable=False, server_default="false")
    is_uploader = Column(Boolean, nullable=False, server_default="false")
    is_reviewer = Column(Boolean, nullable=False, server_default="false")
    updated_by_email = Column(String(255), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
