"""Shared fixtures and mock helpers for unit tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# DB result mock helpers
# ---------------------------------------------------------------------------

def scalar_result(value):
    """Mock execute() result whose .scalar() returns value."""
    r = MagicMock()
    r.scalar.return_value = value
    return r


def one_result(row):
    """Mock execute() result whose .one() returns row."""
    r = MagicMock()
    r.one.return_value = row
    return r


def all_result(rows):
    """Mock execute() result whose .all() returns rows."""
    r = MagicMock()
    r.all.return_value = rows
    return r


def first_result(row):
    """Mock execute() result whose .first() returns row."""
    r = MagicMock()
    r.first.return_value = row
    return r


def scalars_result(objects):
    """Mock execute() result whose .scalars().all() returns objects."""
    r = MagicMock()
    r.scalars.return_value.all.return_value = objects
    return r


# ---------------------------------------------------------------------------
# Fake ORM-like objects (SimpleNamespace satisfies from_attributes=True)
# ---------------------------------------------------------------------------

def make_species(id: int = 1, name: str = "Dog"):
    return SimpleNamespace(id=id, name=name)


def make_cancer_type(id: int = 1, name: str = "Lymphoma", description: str = None):
    return SimpleNamespace(id=id, name=name, description=description)


def make_county(id: int = 1, name: str = "Yolo", fips_code: str = "06113",
                population: int = None, area_sq_miles: float = None):
    return SimpleNamespace(id=id, name=name, fips_code=fips_code,
                           population=population, area_sq_miles=area_sq_miles)


def make_breed(id: int = 1, species_id: int = 1, name: str = "Labrador Retriever"):
    return SimpleNamespace(id=id, species_id=species_id, name=name)
