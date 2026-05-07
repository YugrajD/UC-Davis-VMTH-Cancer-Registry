"""CSV export generator for county-level analysis data."""

import csv
import io

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import CalEnviroScreen, County, Patient
from app.services.review_filter import review_status_sql_in

# ---------------------------------------------------------------------------
# Static data (mirrored from frontend TypeScript sources)
# ---------------------------------------------------------------------------

# Pesticide data for 16 counties (from pesticideData.ts)
# Keys: lbs_per_sq_mile, lbs_applied_total, top_pesticide_class,
#        fumigants, herbicides, insecticides, fungicides, 2015-2019

_PESTICIDE_RAW: dict[str, dict] = {
    "San Joaquin":  {"lbs_per_sq_mile": 312, "lbs_applied_total": 14060000, "top_class": "Fumigants"},
    "Stanislaus":   {"lbs_per_sq_mile": 284, "lbs_applied_total": 10782000, "top_class": "Insecticides"},
    "Colusa":       {"lbs_per_sq_mile": 248, "lbs_applied_total": 2728000,  "top_class": "Herbicides"},
    "Yolo":         {"lbs_per_sq_mile": 196, "lbs_applied_total": 3920000,  "top_class": "Herbicides"},
    "Sacramento":   {"lbs_per_sq_mile": 142, "lbs_applied_total": 7526000,  "top_class": "Fungicides"},
    "Placer":       {"lbs_per_sq_mile": 84,  "lbs_applied_total": 2520000,  "top_class": "Herbicides"},
    "El Dorado":    {"lbs_per_sq_mile": 38,  "lbs_applied_total": 1596000,  "top_class": "Herbicides"},
    "Solano":       {"lbs_per_sq_mile": 118, "lbs_applied_total": 2478000,  "top_class": "Herbicides"},
    "Contra Costa": {"lbs_per_sq_mile": 52,  "lbs_applied_total": 1092000,  "top_class": "Insecticides"},
    "Alameda":      {"lbs_per_sq_mile": 28,  "lbs_applied_total": 504000,   "top_class": "Insecticides"},
    "Glenn":        {"lbs_per_sq_mile": 214, "lbs_applied_total": 2996000,  "top_class": "Herbicides"},
    "Sutter":       {"lbs_per_sq_mile": 192, "lbs_applied_total": 1920000,  "top_class": "Fumigants"},
    "Yuba":         {"lbs_per_sq_mile": 138, "lbs_applied_total": 966000,   "top_class": "Herbicides"},
    "Butte":        {"lbs_per_sq_mile": 106, "lbs_applied_total": 4452000,  "top_class": "Herbicides"},
    "Nevada":       {"lbs_per_sq_mile": 44,  "lbs_applied_total": 1320000,  "top_class": "Herbicides"},
    "Amador":       {"lbs_per_sq_mile": 32,  "lbs_applied_total": 384000,   "top_class": "Herbicides"},
}

_CENTRAL_VALLEY = {
    "San Joaquin", "Stanislaus", "Colusa", "Yolo", "Sacramento",
    "Glenn", "Sutter", "Yuba", "Butte",
}


def _distribute_by_class(total: float, top: str) -> dict[str, float]:
    classes = ["fumigants", "herbicides", "insecticides", "fungicides"]
    top_key = top.lower()
    result: dict[str, float] = {}
    for cls in classes:
        if cls == top_key:
            result[cls] = round(total * 0.45)
        else:
            idx_top = classes.index(top_key)
            if cls == classes[(idx_top + 1) % 4]:
                result[cls] = round(total * 0.22)
            elif cls == classes[(idx_top + 2) % 4]:
                result[cls] = round(total * 0.18)
            else:
                result[cls] = round(total * 0.15)
    return result


def _generate_by_year(avg: float, county: str) -> dict[int, float]:
    is_cv = county in _CENTRAL_VALLEY
    offsets = (
        {2015: -0.06, 2016: -0.03, 2017: 0.0, 2018: 0.04, 2019: 0.08}
        if is_cv
        else {2015: -0.02, 2016: 0.01, 2017: 0.0, 2018: -0.01, 2019: 0.02}
    )
    return {yr: round(avg * (1 + pct)) for yr, pct in offsets.items()}


def _build_pesticide_data() -> dict[str, dict]:
    """Expand raw pesticide data with class breakdown and yearly values."""
    result: dict[str, dict] = {}
    for county, raw in _PESTICIDE_RAW.items():
        by_class = _distribute_by_class(raw["lbs_per_sq_mile"], raw["top_class"])
        by_year = _generate_by_year(raw["lbs_per_sq_mile"], county)
        result[county] = {
            "lbs_per_sq_mile": raw["lbs_per_sq_mile"],
            "total_lbs": raw["lbs_applied_total"],
            "top_class": raw["top_class"],
            "fumigants": by_class["fumigants"],
            "herbicides": by_class["herbicides"],
            "insecticides": by_class["insecticides"],
            "fungicides": by_class["fungicides"],
            2015: by_year[2015],
            2016: by_year[2016],
            2017: by_year[2017],
            2018: by_year[2018],
            2019: by_year[2019],
        }
    return result


PESTICIDE_DATA = _build_pesticide_data()

# Superfund aggregate per county (from superfundData.ts)
SUPERFUND_DATA: dict[str, dict[str, int]] = {
    "Sacramento":   {"total": 3, "active": 2, "remediated": 1},
    "San Joaquin":  {"total": 2, "active": 1, "remediated": 1},
    "Stanislaus":   {"total": 2, "active": 1, "remediated": 0},
    "Yolo":         {"total": 1, "active": 0, "remediated": 1},
    "Placer":       {"total": 1, "active": 1, "remediated": 0},
    "El Dorado":    {"total": 1, "active": 0, "remediated": 1},
    "Contra Costa": {"total": 3, "active": 1, "remediated": 2},
    "Solano":       {"total": 2, "active": 1, "remediated": 1},
    "Alameda":      {"total": 2, "active": 1, "remediated": 1},
    "Butte":        {"total": 1, "active": 0, "remediated": 1},
    "Sutter":       {"total": 1, "active": 0, "remediated": 0},
    "Yuba":         {"total": 1, "active": 1, "remediated": 0},
    "Colusa":       {"total": 1, "active": 0, "remediated": 0},
}

# Human cancer rate: "All Cancer Sites", "Both Sexes" (first occurrence per
# county from humanCancerRates.ts — rate per 100k, age-adjusted)
HUMAN_CANCER_RATE: dict[str, float] = {
    "Alameda": 371.6,
    "Alpine": 268.5,
    "Amador": 427.1,
    "Butte": 483.2,
    "Calaveras": 416.0,
    "Colusa": 373.6,
    "Contra Costa": 409.6,
    "Del Norte": 342.4,
    "El Dorado": 423.8,
    "Fresno": 390.1,
    "Glenn": 487.2,
    "Humboldt": 440.5,
    "Imperial": 363.7,
    "Inyo": 345.4,
    "Kern": 402.5,
    "Kings": 364.6,
    "Lake": 419.7,
    "Lassen": 331.2,
    "Los Angeles": 369.1,
    "Madera": 393.9,
    "Marin": 443.4,
    "Mariposa": 402.9,
    "Mendocino": 404.7,
    "Merced": 378.9,
    "Modoc": 290.7,
    "Mono": 291.4,
    "Monterey": 373.8,
    "Napa": 416.2,
    "Nevada": 400.9,
    "Orange": 407.8,
    "Placer": 436.1,
    "Plumas": 360.1,
    "Riverside": 397.8,
    "Sacramento": 406.9,
    "San Benito": 396.9,
    "San Bernardino": 398.6,
    "San Diego": 426.1,
    "San Francisco": 384.5,
    "San Joaquin": 396.2,
    "San Luis Obispo": 455.6,
    "San Mateo": 398.8,
    "Santa Barbara": 452.3,
    "Santa Clara": 384.0,
    "Santa Cruz": 463.9,
    "Shasta": 461.4,
    "Sierra": 259.6,
    "Siskiyou": 357.4,
    "Solano": 417.6,
    "Sonoma": 435.6,
    "Stanislaus": 423.6,
    "Sutter": 415.3,
    "Tehama": 448.6,
    "Trinity": 300.4,
    "Tulare": 361.7,
    "Tuolumne": 421.2,
    "Ventura": 434.4,
    "Yolo": 413.8,
    "Yuba": 447.3,
}

# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "county_name",
    "fips_code",
    "vmth_cancer_cases",
    "ces_score",
    "pollution_burden",
    "ozone",
    "pm25",
    "diesel_pm",
    "ces_pesticides",
    "toxic_releases",
    "traffic",
    "drinking_water",
    "lead",
    "cleanup_sites",
    "groundwater_threats",
    "hazardous_waste",
    "solid_waste",
    "impaired_water",
    "pop_characteristics",
    "asthma",
    "low_birth_weight",
    "cardiovascular",
    "poverty",
    "unemployment",
    "housing_burden",
    "education",
    "linguistic_isolation",
    "pesticide_lbs_per_sq_mile",
    "pesticide_total_lbs",
    "pesticide_top_class",
    "pesticide_fumigants",
    "pesticide_herbicides",
    "pesticide_insecticides",
    "pesticide_fungicides",
    "pesticide_2015",
    "pesticide_2016",
    "pesticide_2017",
    "pesticide_2018",
    "pesticide_2019",
    "superfund_total",
    "superfund_active",
    "superfund_remediated",
    "human_cancer_rate_all_sites",
]

CES_INDICATOR_COLS = [
    "ces_score", "pollution_burden", "ozone", "pm25", "diesel_pm",
    "pesticides", "toxic_releases", "traffic", "drinking_water", "lead",
    "cleanup_sites", "groundwater_threats", "hazardous_waste", "solid_waste",
    "impaired_water", "pop_characteristics", "asthma", "low_birth_weight",
    "cardiovascular", "poverty", "unemployment", "housing_burden",
    "education", "linguistic_isolation",
]


async def generate_county_export_csv(db: AsyncSession) -> str:
    """Build a CSV string with one row per CA county (58 rows)."""

    # 1. All counties
    counties_result = await db.execute(
        select(County.name, County.fips_code).order_by(County.name)
    )
    counties = counties_result.all()

    # 2. VMTH cancer cases by county
    visible = review_status_sql_in()
    vmth_query = text(f"""
        SELECT c.name AS county_name, COUNT(DISTINCT p.id) AS cases
        FROM patients p
        JOIN counties c ON c.id = p.county_id
        JOIN case_diagnoses cd ON cd.patient_id = p.id
        WHERE p.data_source = 'petbert'
          AND cd.review_status IN {visible}
        GROUP BY c.name
    """)
    vmth_result = await db.execute(vmth_query)
    vmth_by_county: dict[str, int] = {
        row.county_name: row.cases for row in vmth_result.all()
    }

    # 3. CalEnviroScreen data
    ces_result = await db.execute(
        select(
            County.name,
            CalEnviroScreen.ces_score,
            CalEnviroScreen.pollution_burden,
            CalEnviroScreen.ozone,
            CalEnviroScreen.pm25,
            CalEnviroScreen.diesel_pm,
            CalEnviroScreen.pesticides,
            CalEnviroScreen.toxic_releases,
            CalEnviroScreen.traffic,
            CalEnviroScreen.drinking_water,
            CalEnviroScreen.lead,
            CalEnviroScreen.cleanup_sites,
            CalEnviroScreen.groundwater_threats,
            CalEnviroScreen.hazardous_waste,
            CalEnviroScreen.solid_waste,
            CalEnviroScreen.impaired_water,
            CalEnviroScreen.pop_characteristics,
            CalEnviroScreen.asthma,
            CalEnviroScreen.low_birth_weight,
            CalEnviroScreen.cardiovascular,
            CalEnviroScreen.poverty,
            CalEnviroScreen.unemployment,
            CalEnviroScreen.housing_burden,
            CalEnviroScreen.education,
            CalEnviroScreen.linguistic_isolation,
        )
        .join(County, County.id == CalEnviroScreen.county_id)
    )
    ces_by_county: dict[str, dict] = {}
    for row in ces_result.all():
        ces_by_county[row[0]] = {
            col: (float(row[i + 1]) if row[i + 1] is not None else "")
            for i, col in enumerate(CES_INDICATOR_COLS)
        }

    # 4. Build CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for county_name, fips_code in counties:
        ces = ces_by_county.get(county_name, {})
        pest = PESTICIDE_DATA.get(county_name, {})
        sf = SUPERFUND_DATA.get(county_name, {})
        human_rate = HUMAN_CANCER_RATE.get(county_name, "")

        row: dict = {
            "county_name": county_name,
            "fips_code": fips_code,
            "vmth_cancer_cases": vmth_by_county.get(county_name, 0),
        }

        # CES indicators
        for col in CES_INDICATOR_COLS:
            csv_col = col if col.startswith("ces_") else col
            # Map CES column names to CSV column names
            if col == "pesticides":
                csv_col = "ces_pesticides"
            row[csv_col] = ces.get(col, "")

        # Pesticide data
        row["pesticide_lbs_per_sq_mile"] = pest.get("lbs_per_sq_mile", "")
        row["pesticide_total_lbs"] = pest.get("total_lbs", "")
        row["pesticide_top_class"] = pest.get("top_class", "")
        row["pesticide_fumigants"] = pest.get("fumigants", "")
        row["pesticide_herbicides"] = pest.get("herbicides", "")
        row["pesticide_insecticides"] = pest.get("insecticides", "")
        row["pesticide_fungicides"] = pest.get("fungicides", "")
        row["pesticide_2015"] = pest.get(2015, "")
        row["pesticide_2016"] = pest.get(2016, "")
        row["pesticide_2017"] = pest.get(2017, "")
        row["pesticide_2018"] = pest.get(2018, "")
        row["pesticide_2019"] = pest.get(2019, "")

        # Superfund data
        row["superfund_total"] = sf.get("total", "")
        row["superfund_active"] = sf.get("active", "")
        row["superfund_remediated"] = sf.get("remediated", "")

        # Human cancer rate
        row["human_cancer_rate_all_sites"] = human_rate

        writer.writerow(row)

    return output.getvalue()
