-- 008_all_ca_counties.sql
-- Expand counties table to all 58 California counties.
-- Adds is_catchment flag to distinguish the original 16 UCD catchment counties.

ALTER TABLE counties ADD COLUMN IF NOT EXISTS is_catchment BOOLEAN NOT NULL DEFAULT FALSE;

-- Mark the original 16 as catchment
UPDATE counties SET is_catchment = TRUE
WHERE name IN (
    'Sacramento', 'Yolo', 'Solano', 'Placer', 'El Dorado',
    'San Joaquin', 'Contra Costa', 'Alameda', 'Stanislaus',
    'Sutter', 'Yuba', 'Nevada', 'Amador', 'Butte', 'Colusa', 'Glenn'
);

-- Insert remaining 42 California counties
INSERT INTO counties (name, fips_code, population, area_sq_miles, is_catchment) VALUES
    ('Alpine',          '06003',   1204,    739.00, FALSE),
    ('Calaveras',       '06009',  45905,   1020.07, FALSE),
    ('Del Norte',       '06015',  27812,   1008.33, FALSE),
    ('Fresno',          '06019', 999101,   5958.45, FALSE),
    ('Humboldt',        '06023', 136310,   3573.00, FALSE),
    ('Imperial',        '06025', 179702,   4175.00, FALSE),
    ('Inyo',            '06027',  18039,  10192.00, FALSE),
    ('Kern',            '06029', 900202,   8142.00, FALSE),
    ('Kings',           '06031', 152940,   1389.00, FALSE),
    ('Lake',            '06033',  68163,   1258.00, FALSE),
    ('Lassen',          '06035',  30573,   4558.00, FALSE),
    ('Los Angeles',     '06037', 10014009, 4058.00, FALSE),
    ('Madera',          '06039', 157327,   2137.00, FALSE),
    ('Marin',           '06041', 262321,    520.00, FALSE),
    ('Mariposa',        '06043',  17131,   1451.00, FALSE),
    ('Mendocino',       '06045',  91601,   3509.00, FALSE),
    ('Merced',          '06047', 281202,   1929.00, FALSE),
    ('Modoc',           '06049',   8700,   3944.00, FALSE),
    ('Mono',            '06051',  14444,   3044.00, FALSE),
    ('Monterey',        '06053', 434061,   3322.00, FALSE),
    ('Napa',            '06055', 138019,    754.00, FALSE),
    ('Orange',          '06059', 3186989,   791.00, FALSE),
    ('Plumas',          '06063',  19790,   2554.00, FALSE),
    ('Riverside',       '06065', 2418185,   7208.00, FALSE),
    ('San Benito',      '06069',  64209,   1389.00, FALSE),
    ('San Bernardino',  '06071', 2181654,  20062.00, FALSE),
    ('San Diego',       '06073', 3298634,   4207.00, FALSE),
    ('San Francisco',   '06075',  873965,     47.00, FALSE),
    ('San Luis Obispo', '06079', 283111,   3299.00, FALSE),
    ('San Mateo',       '06081', 764442,    449.00, FALSE),
    ('Santa Barbara',   '06083', 446499,   2738.00, FALSE),
    ('Santa Clara',     '06085', 1936259,   1291.00, FALSE),
    ('Santa Cruz',      '06087', 270861,    445.00, FALSE),
    ('Shasta',          '06089', 180080,   3786.00, FALSE),
    ('Sierra',          '06091',   3236,    953.00, FALSE),
    ('Siskiyou',        '06093',  44076,   6287.00, FALSE),
    ('Sonoma',          '06097', 488863,   1576.00, FALSE),
    ('Tehama',          '06103',  65084,   2951.00, FALSE),
    ('Trinity',         '06105',  16112,   3179.00, FALSE),
    ('Tulare',          '06107', 466195,   4824.00, FALSE),
    ('Tuolumne',        '06109',  55620,   2236.00, FALSE),
    ('Ventura',         '06111', 843843,   1846.00, FALSE)
ON CONFLICT (name) DO NOTHING;
