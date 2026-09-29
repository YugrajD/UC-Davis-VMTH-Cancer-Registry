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
    birth_date = Column(Date, nullable=True)
    diagnosis_date = Column(Date, nullable=True)
    outcome = Column(String(20), nullable=True)
    # Set when a combined_predictions load codes this case NO_CANCER — a case
    # with zero case_diagnoses rows and this false just hasn't been coded yet.
    # See database/migrations/035_combined_predictions.sql.
    registry_no_cancer = Column(Boolean, nullable=False, server_default="false")
    registry_no_cancer_source_version = Column(String(80), nullable=True)
    # Set when a review_queue load lists this case with no combined_predictions
    # row yet (queued, not coded).
    registry_awaiting_review = Column(Boolean, nullable=False, server_default="false")

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
    pathology_report_id = Column(Integer, ForeignKey("pathology_reports.id"), nullable=True)
    confidence = Column(Numeric(4, 2), nullable=True)
    prediction_method = Column(String(50), nullable=True)
    source_row_index = Column(Integer, nullable=True)
    diagnosis_index = Column(Integer, nullable=True)
    # The report-mapping generation_id that produced this code (e.g. "gen-20260927T003905Z"),
    # from the worker's source_version — see database/migrations/032_case_diagnosis_source_version.sql
    source_version = Column(String(80), nullable=True)

    # Provenance from a combined_predictions load — see
    # database/migrations/035_combined_predictions.sql and ml/coding/combine.py.
    # code_source: 'manual' (gold) / 'diagnosis' (silver) / 'report' (bronze).
    # source_confidence is free text: a decision-stage name for silver (e.g.
    # "tier1_exact"), a numeric string for bronze, empty for gold — never a
    # column read as a number; the existing `confidence` column above still
    # holds bronze's numeric value for anything that sorts/displays by it.
    # ml_review_status is ML's own raw value (confirmed/auto_accepted/queued),
    # kept alongside our mapped `review_status` below so the UI can tell a
    # specialist's code from a machine's.
    code_source = Column(String(20), nullable=True)
    source_confidence = Column(Text, nullable=True)
    ml_review_status = Column(String(20), nullable=True)

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
    pathology_report = relationship("PathologyReport", back_populates="diagnoses")
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
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    gcs_path = Column(String(1000), nullable=True)
    report_date = Column(Date, nullable=True)
    source_diagnosis = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    patient = relationship("Patient", back_populates="reports")
    diagnoses = relationship("CaseDiagnosis", back_populates="pathology_report")


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
    model_folder = Column(String(255), nullable=True)
    clinic_name = Column(String(255), nullable=True)
    upload_duration_ms = Column(Integer, nullable=True)
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


class RoleRequest(Base):
    """User-submitted requests for uploader/reviewer roles."""
    __tablename__ = "role_requests"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), nullable=False)
    requested_role = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, server_default="pending")
    reason = Column(Text, nullable=True)
    resolved_by_email = Column(String(255), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ExportRequest(Base):
    """User-submitted requests for data export access."""
    __tablename__ = "export_requests"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, server_default="pending")
    reason = Column(Text, nullable=True)
    resolved_by_email = Column(String(255), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


# --- Audit-list / gold review — see database/migrations/033_gold_review.sql ---
# and ml/documentation/audit-list-change-request.md.


class TaxonomyTerm(Base):
    """A (group, term) pair from ml/taxonomy/labels.csv, seeded once (not read
    live — production has no /ml mount). Backs the review screen's code picker
    and validates case_review_codes rows before they're saved or exported."""
    __tablename__ = "taxonomy_terms"

    id = Column(Integer, primary_key=True)
    vet_icd_o_code = Column(String(20), nullable=True)
    taxonomy_group = Column(String(255), nullable=False)
    taxonomy_term = Column(String(255), nullable=False)
    # "Preferred" / "Synonym" / "Related", from labels.csv's `level` column
    # (migration 036). Nullable until the seed is re-run against it.
    term_level = Column(String(20), nullable=True)


class AuditList(Base):
    """One imported audit_list_<list_id>.txt. Only one is ever active — the
    specialist's current worklist; loading a new list flips this one off."""
    __tablename__ = "audit_lists"

    id = Column(Integer, primary_key=True)
    list_id = Column(String(100), nullable=False, unique=True)
    imported_by_email = Column(String(255), nullable=False)
    imported_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    sha256 = Column(String(64), nullable=False)
    case_count = Column(Integer, nullable=False)
    is_active = Column(Boolean, nullable=False, server_default="false")

    cases = relationship("AuditListCase", back_populates="audit_list", cascade="all, delete-orphan")


class AuditListCase(Base):
    """case_id + position within a list, exactly as ML sent it (never
    normalized — the gold export echoes it back verbatim). Kept across every
    list ever imported: export eligibility checks this table's full history,
    since ML refuses the whole gold file if a case_id was never on any list."""
    __tablename__ = "audit_list_cases"

    id = Column(Integer, primary_key=True)
    audit_list_id = Column(Integer, ForeignKey("audit_lists.id", ondelete="CASCADE"), nullable=False)
    case_id = Column(String(100), nullable=False)
    position = Column(Integer, nullable=False)

    audit_list = relationship("AuditList", back_populates="cases")


class GoldExport(Base):
    """One admin-triggered export batch — a gold_<export_id>.csv for one
    reviewer's completed, unlocked reviews."""
    __tablename__ = "gold_exports"

    id = Column(Integer, primary_key=True)
    export_id = Column(String(100), nullable=False, unique=True)
    reviewer_email = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    case_count = Column(Integer, nullable=False)


class CaseReview(Base):
    """One row per case ever reviewed from the audit-list worklist: either
    no_cancer or a complete code set in case_review_codes, never both
    (enforced in the router, not here).

    Editable while locked = false. Exporting sets locked = true and records
    the export in gold_export_id/exported_at — the most recent export this
    case was part of, not a version history; ML's own audit_list_ledger.csv
    is the system of record for prior gold versions. An admin can explicitly
    reopen a locked review; re-exporting after an edit replaces ML's copy
    (audit-list-change-request.md: "re-sending a case replaces its earlier
    review on ML's side")."""
    __tablename__ = "case_reviews"

    id = Column(Integer, primary_key=True)
    case_id = Column(String(100), nullable=False, unique=True)
    no_cancer = Column(Boolean, nullable=False, server_default="false")
    reviewed_by_email = Column(String(255), nullable=False)
    reviewed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    locked = Column(Boolean, nullable=False, server_default="false")
    gold_export_id = Column(Integer, ForeignKey("gold_exports.id", ondelete="SET NULL"), nullable=True)
    exported_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    codes = relationship("CaseReviewCode", back_populates="case_review", cascade="all, delete-orphan")
    gold_export = relationship("GoldExport")


class CaseReviewCode(Base):
    """One code in a case's complete set (only when case_reviews.no_cancer is
    false). (group, term) must exist in taxonomy_terms — validated when the
    review is saved and again at export time, in case the taxonomy changed."""
    __tablename__ = "case_review_codes"

    id = Column(Integer, primary_key=True)
    case_review_id = Column(Integer, ForeignKey("case_reviews.id", ondelete="CASCADE"), nullable=False)
    taxonomy_group = Column(String(255), nullable=False)
    taxonomy_term = Column(String(255), nullable=False)

    case_review = relationship("CaseReview", back_populates="codes")
