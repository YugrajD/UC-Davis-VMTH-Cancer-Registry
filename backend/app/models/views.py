"""SQLAlchemy Table definitions for materialized views (read-only)."""

from sqlalchemy import Column, Integer, MetaData, String, Table

mv_metadata = MetaData()

mv_county_cancer = Table(
    "mv_county_cancer_incidence", mv_metadata,
    Column("county_id", Integer),
    Column("county_name", String),
    Column("cancer_type_id", Integer),
    Column("cancer_type_name", String),
    Column("species_id", Integer),
    Column("species_name", String),
    Column("sex", String),
    Column("year", Integer),
    Column("case_count", Integer),
)

mv_yearly_trends = Table(
    "mv_yearly_trends", mv_metadata,
    Column("year", Integer),
    Column("cancer_type_id", Integer),
    Column("cancer_type_name", String),
    Column("species_id", Integer),
    Column("species_name", String),
    Column("county_id", Integer),
    Column("county_name", String),
    Column("sex", String),
    Column("case_count", Integer),
    Column("deceased_count", Integer),
    Column("alive_count", Integer),
)
