-- 002_lookup_tables.sql
-- Species, breeds, and cancer types lookup tables

CREATE TABLE IF NOT EXISTS species (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS breeds (
    id SERIAL PRIMARY KEY,
    species_id INTEGER NOT NULL REFERENCES species(id),
    name VARCHAR(100) NOT NULL,
    UNIQUE(species_id, name)
);

CREATE TABLE IF NOT EXISTS cancer_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT
);

-- Seed species
INSERT INTO species (name) VALUES
    ('Dog'),
    ('Cat')
ON CONFLICT (name) DO NOTHING;

-- Seed breeds
INSERT INTO breeds (species_id, name) VALUES
    -- Dogs (species_id = 1)
    (1, 'Golden Retriever'),
    (1, 'Labrador Retriever'),
    (1, 'Boxer'),
    (1, 'German Shepherd'),
    (1, 'Rottweiler'),
    (1, 'Bernese Mountain Dog'),
    (1, 'Beagle'),
    (1, 'Poodle'),
    (1, 'Bulldog'),
    (1, 'Mixed Breed Dog'),
    -- Cats (species_id = 2)
    (2, 'Siamese'),
    (2, 'Persian'),
    (2, 'Maine Coon'),
    (2, 'Domestic Shorthair'),
    (2, 'Domestic Longhair'),
    (2, 'Bengal')
ON CONFLICT DO NOTHING;

-- Seed cancer types
INSERT INTO cancer_types (name, description) VALUES
    ('Lymphoma', 'Cancer of the lymphatic system; one of the most common cancers in dogs and cats'),
    ('Mast Cell Tumor', 'Tumor arising from mast cells; most common skin tumor in dogs'),
    ('Osteosarcoma', 'Aggressive bone cancer; common in large breed dogs'),
    ('Hemangiosarcoma', 'Cancer of blood vessel walls; common in dogs, especially Golden Retrievers and German Shepherds'),
    ('Melanoma', 'Cancer of melanocytes; common in oral cavity of dogs'),
    ('Squamous Cell Carcinoma', 'Cancer of squamous epithelial cells; common in cats and horses'),
    ('Fibrosarcoma', 'Soft tissue sarcoma; associated with injection sites in cats'),
    ('Transitional Cell Carcinoma', 'Bladder cancer; common in certain dog breeds')
ON CONFLICT (name) DO NOTHING;
