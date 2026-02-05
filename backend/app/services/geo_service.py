"""GeoPandas/PostGIS geospatial query service."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


async def get_county_centroids(db: AsyncSession) -> list[dict]:
    """Get county centroids for map positioning."""
    query = text("""
        SELECT
            id, name, fips_code,
            ST_X(ST_Centroid(geom)) AS lng,
            ST_Y(ST_Centroid(geom)) AS lat
        FROM counties
        WHERE geom IS NOT NULL
    """)
    result = await db.execute(query)
    return [
        {"id": r.id, "name": r.name, "fips_code": r.fips_code,
         "lat": r.lat, "lng": r.lng}
        for r in result.all()
    ]


async def get_county_geojson(db: AsyncSession, county_id: int) -> dict | None:
    """Get a single county's GeoJSON geometry."""
    query = text("""
        SELECT ST_AsGeoJSON(geom)::json AS geometry
        FROM counties WHERE id = :county_id
    """)
    result = await db.execute(query, {"county_id": county_id})
    row = result.first()
    return row.geometry if row else None
