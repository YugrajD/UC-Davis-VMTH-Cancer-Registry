-- 003_counties.sql
-- Counties table with PostGIS geometry for the UC Davis catchment area

CREATE TABLE IF NOT EXISTS counties (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    fips_code VARCHAR(5) NOT NULL UNIQUE,
    geom GEOMETRY(MULTIPOLYGON, 4326),
    population INTEGER,
    area_sq_miles NUMERIC(10, 2)
);

CREATE INDEX IF NOT EXISTS idx_counties_geom ON counties USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_counties_name ON counties (name);

-- Insert 16 Northern CA counties in UCD catchment area (geometry loaded by seed script)
INSERT INTO counties (name, fips_code, population, area_sq_miles) VALUES
    ('Sacramento', '06067', 1585055, 994.22),
    ('Yolo', '06113', 220500, 1012.07),
    ('Solano', '06095', 453491, 829.19),
    ('Placer', '06061', 404739, 1503.09),
    ('El Dorado', '06017', 192843, 1711.03),
    ('San Joaquin', '06077', 779233, 1399.41),
    ('Contra Costa', '06013', 1153526, 720.07),
    ('Alameda', '06001', 1671329, 738.55),
    ('Stanislaus', '06099', 552878, 1494.77),
    ('Sutter', '06101', 99633, 602.42),
    ('Yuba', '06115', 81575, 631.76),
    ('Nevada', '06057', 102241, 958.38),
    ('Amador', '06005', 40474, 605.42),
    ('Butte', '06007', 211632, 1640.11),
    ('Colusa', '06011', 21917, 1150.72),
    ('Glenn', '06021', 28917, 1314.67)
ON CONFLICT (name) DO NOTHING;
