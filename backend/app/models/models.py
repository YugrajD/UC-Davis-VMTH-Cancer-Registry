"""SQLAlchemy + GeoAlchemy2 models for the VMTH Cancer Registry."""

from sqlalchemy import (
    Column, Integer, String, Numeric, Date, Text, ForeignKey, CheckConstraint
)
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


class County(Base):
    __tablename__ = "counties"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    fips_code = Column(String(5), nullable=False, unique=True)
    geom = Column(Geometry("MULTIPOLYGON", srid=4326))
    population = Column(Integer)
    area_sq_miles = Column(Numeric(10, 2))

    patients = relationship("Patient", back_populates="county")
    cases = relationship("CancerCase", back_populates="county")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    species_id = Column(Integer, ForeignKey("species.id"), nullable=False)
    breed_id = Column(Integer, ForeignKey("breeds.id"), nullable=False)
    sex = Column(String(10), nullable=False)
    age_years = Column(Numeric(5, 1), nullable=False)
    weight_kg = Column(Numeric(6, 2))
    county_id = Column(Integer, ForeignKey("counties.id"), nullable=False)
    registered_date = Column(Date, nullable=False)

    species = relationship("Species", back_populates="patients")
    breed = relationship("Breed", back_populates="patients")
    county = relationship("County", back_populates="patients")
    cases = relationship("CancerCase", back_populates="patient")


class CancerCase(Base):
    __tablename__ = "cancer_cases"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    cancer_type_id = Column(Integer, ForeignKey("cancer_types.id"), nullable=False)
    diagnosis_date = Column(Date, nullable=False)
    stage = Column(String(5))
    outcome = Column(String(20))
    county_id = Column(Integer, ForeignKey("counties.id"), nullable=False)

    patient = relationship("Patient", back_populates="cases")
    cancer_type = relationship("CancerType", back_populates="cases")
    county = relationship("County", back_populates="cases")
    reports = relationship("PathologyReport", back_populates="case")


class PathologyReport(Base):
    __tablename__ = "pathology_reports"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cancer_cases.id"), nullable=False)
    report_text = Column(Text, nullable=False)
    classification = Column(String(100))
    confidence_score = Column(Numeric(5, 4))
    report_date = Column(Date, nullable=False)

    case = relationship("CancerCase", back_populates="reports")
