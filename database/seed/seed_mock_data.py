#!/usr/bin/env python3
"""
Generate and load ~5000 mock cancer cases into the VMTH Cancer Registry database.
Includes realistic distributions for species, breeds, counties, and cancer types.
"""

import os
import sys
import random
import json
from datetime import date, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:postgres@localhost:5432/vmth_cancer"
)

random.seed(42)

# -------------------------------------------------------------------
# Distribution weights
# -------------------------------------------------------------------

# County populations used for weighting case distribution
COUNTY_WEIGHTS = {
    "Sacramento": 0.22,
    "Yolo": 0.06,
    "Solano": 0.08,
    "Placer": 0.09,
    "El Dorado": 0.05,
    "San Joaquin": 0.10,
    "Contra Costa": 0.10,
    "Alameda": 0.08,
    "Stanislaus": 0.06,
    "Sutter": 0.03,
    "Yuba": 0.02,
    "Nevada": 0.03,
    "Amador": 0.01,
    "Butte": 0.04,
    "Colusa": 0.01,
    "Glenn": 0.02,
}

# Species distribution
SPECIES_WEIGHTS = {
    "Dog": 0.65,
    "Cat": 0.35,
}

# Cancer type distributions per species (relative weights)
CANCER_BY_SPECIES = {
    "Dog": {
        "Lymphoma": 25,
        "Mast Cell Tumor": 22,
        "Osteosarcoma": 12,
        "Hemangiosarcoma": 15,
        "Melanoma": 10,
        "Squamous Cell Carcinoma": 5,
        "Fibrosarcoma": 5,
        "Transitional Cell Carcinoma": 6,
    },
    "Cat": {
        "Lymphoma": 35,
        "Squamous Cell Carcinoma": 20,
        "Fibrosarcoma": 18,
        "Mast Cell Tumor": 12,
        "Melanoma": 5,
        "Osteosarcoma": 5,
        "Hemangiosarcoma": 3,
        "Transitional Cell Carcinoma": 2,
    },
}

# Age ranges by species (mean, std)
AGE_PARAMS = {
    "Dog": (8.5, 3.0),
    "Cat": (10.0, 3.5),
}

# Weight ranges by species (mean, std) in kg
WEIGHT_PARAMS = {
    "Dog": (25.0, 12.0),
    "Cat": (4.5, 1.5),
}

SEXES = ["Male", "Female", "Neutered Male", "Spayed Female"]
SEX_WEIGHTS = [0.15, 0.15, 0.35, 0.35]

STAGES = ["I", "II", "III", "IV"]
STAGE_WEIGHTS = [0.25, 0.35, 0.25, 0.15]

OUTCOMES = ["alive", "deceased", "unknown"]
OUTCOME_WEIGHTS = [0.40, 0.45, 0.15]

# Pathology report templates
REPORT_TEMPLATES = {
    "Lymphoma": [
        "Histopathologic examination of the {site} reveals a diffuse infiltrate of neoplastic round cells consistent with {subtype} lymphoma. The neoplastic lymphocytes are large with prominent nucleoli and frequent mitotic figures ({mitotic}/10 HPF). Immunohistochemistry confirms {marker} phenotype. Diagnosis: {grade} lymphoma.",
        "Biopsy of the {site} demonstrates effacement of normal architecture by a monomorphic population of neoplastic lymphoid cells. Cells exhibit round to oval nuclei with finely stippled chromatin. Flow cytometry reveals {marker} immunophenotype. Grade: {grade}. Consistent with multicentric lymphoma.",
        "Fine needle aspirate of the {site} shows a predominant population of intermediate to large lymphoid cells with basophilic cytoplasm. Numerous mitoses observed. Cytologic interpretation: {subtype} lymphoma, {grade}.",
    ],
    "Mast Cell Tumor": [
        "Excisional biopsy reveals a dermal neoplasm composed of round cells with metachromatic cytoplasmic granules consistent with mast cells. Patnaik grade {patnaik}. Mitotic index: {mitotic}/10 HPF. Surgical margins are {margins}. Ki-67 index: {ki67}%. Diagnosis: Mast cell tumor, grade {patnaik}.",
        "Histopathology of the {site} mass demonstrates a well-circumscribed to infiltrative neoplasm of mast cells. Cellular morphology ranges from well-differentiated to pleomorphic. Kiupel grade: {kiupel}. {margins} margins. c-KIT pattern: {ckit}.",
        "Punch biopsy of cutaneous mass from {site}. Dermis expanded by sheets of round cells with moderate amounts of granular cytoplasm. Toluidine blue stain highlights metachromatic granules. Grade {patnaik} (Patnaik). Margins {margins}.",
    ],
    "Osteosarcoma": [
        "Biopsy of the {bone} reveals a highly cellular neoplasm producing osteoid matrix. Neoplastic osteoblasts display marked anisocytosis and anisokaryosis with {mitotic} mitotic figures per 10 HPF. Areas of necrosis comprising approximately {necrosis}% of the sample. Diagnosis: Osteosarcoma, {subtype} variant.",
        "Histopathologic examination of the {bone} mass demonstrates a malignant mesenchymal neoplasm with osteoid production. Cells are spindle to polygonal with hyperchromatic nuclei. {subtype} pattern predominates. Alkaline phosphatase staining is strongly positive. Consistent with appendicular osteosarcoma.",
        "Core biopsy from {bone} lesion shows malignant bone-forming tumor. Marked cellular pleomorphism with tumor giant cells. Matrix is predominantly {subtype}. High mitotic rate ({mitotic}/10 HPF). Diagnosis: High-grade osteosarcoma.",
    ],
    "Hemangiosarcoma": [
        "Splenectomy specimen reveals a {size} cm hemorrhagic mass. Histopathology demonstrates a malignant vascular neoplasm with irregular, anastomosing channels lined by pleomorphic endothelial cells. Mitotic index: {mitotic}/10 HPF. Immunohistochemistry: CD31+, vWF+. Diagnosis: Hemangiosarcoma of the spleen.",
        "Biopsy of {organ} mass shows a malignant neoplasm forming vascular channels and solid sheets of spindle cells. Frequent erythrophagocytosis noted. Areas of hemorrhage and necrosis present. Factor VIII-related antigen positive. Consistent with hemangiosarcoma.",
        "Examination of the {organ} tissue reveals a poorly differentiated malignant vascular tumor. Neoplastic cells form irregular vascular spaces filled with blood. High mitotic rate ({mitotic}/10 HPF). CD31 immunoreactive. Stage {stage} hemangiosarcoma.",
    ],
    "Melanoma": [
        "Biopsy of the {site} mass reveals a malignant melanocytic neoplasm. Tumor cells contain variable amounts of melanin pigment. Nuclear atypia is {atypia} with {mitotic} mitotic figures per 10 HPF. Melan-A and PNL2 immunopositive. Diagnosis: Malignant melanoma, {site}.",
        "Histopathologic examination demonstrates a densely cellular neoplasm composed of epithelioid to spindle-shaped melanocytes. Junctional activity present. Pigmentation is {pigmentation}. Mitotic index {mitotic}/10 HPF. S-100 protein positive. Oral malignant melanoma.",
        "Excisional biopsy of {site} lesion shows an invasive melanocytic neoplasm. Cells arranged in nests and sheets with variable melanin content. Nuclear pleomorphism is marked. Ki-67 index: {ki67}%. Margins: {margins}. Diagnosis: Melanoma.",
    ],
    "Squamous Cell Carcinoma": [
        "Biopsy of the {site} reveals nests and cords of neoplastic squamous epithelial cells invading the underlying stroma. Keratin pearl formation is {keratin}. Cellular atypia is {atypia}. Mitotic figures: {mitotic}/10 HPF. Diagnosis: Squamous cell carcinoma, {differentiation} differentiated.",
        "Histopathology of the {site} mass demonstrates an invasive epithelial neoplasm with squamous differentiation. Tumor cells form irregular islands with central keratinization. Desmoplastic stromal response noted. {differentiation} grade squamous cell carcinoma.",
        "Incisional biopsy from {site}: Proliferation of atypical squamous epithelium with invasion through the basement membrane. Individual cell keratinization and intercellular bridges observed. Solar elastosis in surrounding dermis. Diagnosis: SCC, {differentiation} differentiated.",
    ],
    "Fibrosarcoma": [
        "Histopathologic examination of the {site} mass reveals a malignant mesenchymal neoplasm composed of interlacing bundles of spindle cells producing collagen. Mitotic index: {mitotic}/10 HPF. Margins are {margins}. Vimentin positive, S-100 negative. Diagnosis: Fibrosarcoma, grade {grade}.",
        "Biopsy demonstrates a densely cellular spindle cell neoplasm arranged in a herringbone pattern. Moderate to marked cellular atypia with {mitotic} mitotic figures per 10 HPF. Consistent with {subtype} fibrosarcoma. {margins} margins.",
        "Excision of subcutaneous mass from {site}. Histopathology reveals a well-demarcated but non-encapsulated spindle cell sarcoma. Cells produce abundant collagenous matrix. Low to moderate mitotic rate. Diagnosis: Low-grade fibrosarcoma.",
    ],
    "Transitional Cell Carcinoma": [
        "Cystoscopic biopsy of the urinary bladder reveals a papillary neoplasm composed of multilayered transitional epithelium with loss of polarity and cellular atypia. Invasion into the lamina propria is {invasion}. Mitotic figures: {mitotic}/10 HPF. Diagnosis: Transitional cell carcinoma, grade {grade}.",
        "Bladder wall biopsy demonstrates an invasive urothelial neoplasm. Cells arranged in nests and trabeculae with moderate nuclear pleomorphism. {invasion} invasion into muscularis. Uroplakin III positive. Consistent with high-grade TCC.",
        "Histopathology of the trigone biopsy reveals a papillary transitional cell neoplasm with {invasion} invasion. Moderate cellular atypia and {mitotic} mitotic figures per 10 HPF. Urine cytology correlation supports diagnosis of transitional cell carcinoma.",
    ],
}

# Fill-in values for templates
TEMPLATE_FILLS = {
    "site": ["submandibular lymph node", "prescapular lymph node", "mesenteric lymph node",
             "popliteal lymph node", "liver", "spleen", "skin", "oral cavity", "nasal cavity",
             "digit", "ear pinna", "ventral abdomen", "inguinal region", "thoracic wall",
             "left forelimb", "right hindlimb", "perianal region", "gingiva"],
    "subtype": ["B-cell", "T-cell", "large cell", "small cell", "immunoblastic",
                "lymphoblastic", "osteoblastic", "chondroblastic", "fibroblastic",
                "telangiectatic", "vaccine-associated", "peripheral nerve sheath"],
    "marker": ["CD20+ B-cell", "CD3+ T-cell", "CD79a+ B-cell", "CD4+ T-cell"],
    "grade": ["low", "intermediate", "high"],
    "mitotic": ["2", "5", "8", "12", "18", "25", "32", "45"],
    "patnaik": ["I", "II", "III"],
    "kiupel": ["low", "high"],
    "margins": ["clean (>3mm)", "narrow (<1mm)", "incomplete/dirty", "clean (>5mm)"],
    "ki67": ["5", "12", "18", "25", "35", "48"],
    "ckit": ["perimembranous (pattern I)", "focal cytoplasmic (pattern II)", "diffuse cytoplasmic (pattern III)"],
    "bone": ["distal radius", "proximal humerus", "distal femur", "proximal tibia", "distal tibia"],
    "necrosis": ["10", "20", "30", "50"],
    "size": ["3.5", "5.2", "7.8", "10.3", "12.1"],
    "organ": ["spleen", "liver", "right atrium", "skin", "pericardium"],
    "stage": ["I", "II", "III"],
    "atypia": ["mild", "moderate", "marked", "severe"],
    "pigmentation": ["heavy", "moderate", "scant", "amelanotic"],
    "keratin": ["prominent", "moderate", "minimal", "absent"],
    "differentiation": ["well", "moderately", "poorly"],
    "invasion": ["present", "absent", "superficial", "deep"],
}


def weighted_choice(choices_weights: dict) -> str:
    items = list(choices_weights.keys())
    weights = list(choices_weights.values())
    return random.choices(items, weights=weights, k=1)[0]


def generate_report(cancer_type: str) -> str:
    templates = REPORT_TEMPLATES.get(cancer_type, REPORT_TEMPLATES["Lymphoma"])
    template = random.choice(templates)
    result = template
    for key, values in TEMPLATE_FILLS.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def run():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    print("Loading lookup data...")

    # Fetch species
    cur.execute("SELECT id, name FROM species")
    species_map = {name: id_ for id_, name in cur.fetchall()}

    # Fetch breeds grouped by species
    cur.execute("SELECT b.id, b.name, s.name FROM breeds b JOIN species s ON b.species_id = s.id")
    breeds_by_species = {}
    breed_map = {}
    for bid, bname, sname in cur.fetchall():
        breeds_by_species.setdefault(sname, []).append(bid)
        breed_map[bid] = bname

    # Fetch cancer types
    cur.execute("SELECT id, name FROM cancer_types")
    cancer_type_map = {name: id_ for id_, name in cur.fetchall()}

    # Fetch counties
    cur.execute("SELECT id, name FROM counties")
    county_map = {name: id_ for id_, name in cur.fetchall()}

    # -------------------------------------------------------------------
    # Generate patients and cases
    # -------------------------------------------------------------------
    total_cases = 5000
    print(f"Generating {total_cases} cancer cases...")

    patient_rows = []
    case_rows = []
    report_rows = []

    patient_id = 0
    case_id = 0
    report_count = 0

    for i in range(total_cases):
        # Pick species
        species_name = weighted_choice(SPECIES_WEIGHTS)
        species_id = species_map[species_name]

        # Pick breed
        breed_id = random.choice(breeds_by_species[species_name])

        # Pick sex
        sex = random.choices(SEXES, weights=SEX_WEIGHTS, k=1)[0]

        # Generate age
        mean_age, std_age = AGE_PARAMS[species_name]
        age = max(0.5, round(random.gauss(mean_age, std_age), 1))

        # Generate weight
        mean_w, std_w = WEIGHT_PARAMS[species_name]
        weight = max(0.1, round(random.gauss(mean_w, std_w), 2))

        # Pick county
        county_name = weighted_choice(COUNTY_WEIGHTS)
        county_id = county_map[county_name]

        # Diagnosis date (1995-2024), slight upward trend over 30 years
        years = list(range(1995, 2025))
        weights_30 = [1.0 + 0.15 * i for i in range(30)]  # gradual increase
        year = random.choices(years, weights=weights_30, k=1)[0]
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        diagnosis_date = date(year, month, day)
        registered_date = diagnosis_date - timedelta(days=random.randint(0, 60))

        # Pick cancer type based on species
        cancer_name = weighted_choice(CANCER_BY_SPECIES[species_name])
        cancer_type_id = cancer_type_map[cancer_name]

        # Stage and outcome
        stage = random.choices(STAGES, weights=STAGE_WEIGHTS, k=1)[0]
        outcome = random.choices(OUTCOMES, weights=OUTCOME_WEIGHTS, k=1)[0]

        patient_id += 1
        case_id += 1

        patient_rows.append((
            patient_id, species_id, breed_id, sex, age, weight,
            county_id, registered_date
        ))

        case_rows.append((
            case_id, patient_id, cancer_type_id, diagnosis_date,
            stage, outcome, county_id
        ))

        # Generate pathology report for ~10% of cases (= ~500 reports)
        if random.random() < 0.10:
            report_count += 1
            report_text = generate_report(cancer_name)
            confidence = round(random.uniform(0.75, 0.99), 4)
            report_date = diagnosis_date + timedelta(days=random.randint(1, 14))
            report_rows.append((
                case_id, report_text, cancer_name, confidence, report_date
            ))

    # -------------------------------------------------------------------
    # Bulk insert
    # -------------------------------------------------------------------
    print(f"Inserting {len(patient_rows)} patients...")
    execute_values(
        cur,
        """INSERT INTO patients (id, species_id, breed_id, sex, age_years, weight_kg,
                                 county_id, registered_date)
           VALUES %s ON CONFLICT DO NOTHING""",
        patient_rows
    )

    print(f"Inserting {len(case_rows)} cancer cases...")
    execute_values(
        cur,
        """INSERT INTO cancer_cases (id, patient_id, cancer_type_id, diagnosis_date,
                                     stage, outcome, county_id)
           VALUES %s ON CONFLICT DO NOTHING""",
        case_rows
    )

    print(f"Inserting {len(report_rows)} pathology reports...")
    execute_values(
        cur,
        """INSERT INTO pathology_reports (case_id, report_text, classification,
                                          confidence_score, report_date)
           VALUES %s ON CONFLICT DO NOTHING""",
        report_rows
    )

    # Reset sequences
    cur.execute("SELECT setval('patients_id_seq', (SELECT MAX(id) FROM patients))")
    cur.execute("SELECT setval('cancer_cases_id_seq', (SELECT MAX(id) FROM cancer_cases))")

    # Create materialized views (one statement at a time)
    print("Creating materialized views...")
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS mv_county_cancer_incidence CASCADE")
    cur.execute("""
        CREATE MATERIALIZED VIEW mv_county_cancer_incidence AS
        SELECT cc.county_id, co.name AS county_name, ct.id AS cancer_type_id,
               ct.name AS cancer_type_name,
               EXTRACT(YEAR FROM cc.diagnosis_date)::INTEGER AS year,
               COUNT(*) AS case_count, s.name AS species_name
        FROM cancer_cases cc
        JOIN counties co ON cc.county_id = co.id
        JOIN cancer_types ct ON cc.cancer_type_id = ct.id
        JOIN patients p ON cc.patient_id = p.id
        JOIN species s ON p.species_id = s.id
        GROUP BY cc.county_id, co.name, ct.id, ct.name,
                 EXTRACT(YEAR FROM cc.diagnosis_date), s.name
    """)
    print("  mv_county_cancer_incidence created.")

    cur.execute("DROP MATERIALIZED VIEW IF EXISTS mv_yearly_trends CASCADE")
    cur.execute("""
        CREATE MATERIALIZED VIEW mv_yearly_trends AS
        SELECT EXTRACT(YEAR FROM cc.diagnosis_date)::INTEGER AS year,
               ct.id AS cancer_type_id, ct.name AS cancer_type_name,
               s.id AS species_id, s.name AS species_name,
               COUNT(*) AS case_count,
               COUNT(*) FILTER (WHERE cc.outcome = 'deceased') AS deceased_count,
               COUNT(*) FILTER (WHERE cc.outcome = 'alive') AS alive_count
        FROM cancer_cases cc
        JOIN cancer_types ct ON cc.cancer_type_id = ct.id
        JOIN patients p ON cc.patient_id = p.id
        JOIN species s ON p.species_id = s.id
        GROUP BY EXTRACT(YEAR FROM cc.diagnosis_date), ct.id, ct.name, s.id, s.name
    """)
    print("  mv_yearly_trends created.")

    # Load county boundaries
    print("Loading county boundaries...")
    sys.path.insert(0, os.path.dirname(__file__))
    from county_boundaries import COUNTY_GEOMETRIES
    import json as _json
    for county_name, geojson in COUNTY_GEOMETRIES.items():
        geom_json = _json.dumps(geojson)
        cur.execute(
            "UPDATE counties SET geom = ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326) WHERE name = %s",
            (geom_json, county_name)
        )
    print("County boundaries loaded.")

    conn.commit()
    print(f"Done! Inserted {len(patient_rows)} patients, {len(case_rows)} cases, {len(report_rows)} reports.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    run()
