#!/usr/bin/env python3
"""
Generate frontend/src/data/pesticideData.ts from CDPR PUR county-level data.

Downloads chemical-subtotals-by-county files for 2016-2023, joins against
a built-in chemical → use-type lookup, and outputs a TypeScript module with
all 58 CA counties and per-year, per-class lbs/sq-mi data.

Usage:
    python3 database/seed/generate_pesticide_data.py
"""

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

BASE = "https://files.cdpr.ca.gov/pub/outgoing/pur/data"

YEARS = list(range(2016, 2024))

# Try lowercase first (matches confirmed 2019 URL), then uppercase PUR
def county_subtotals_url(year: int) -> list[str]:
    # 2016-2018 use a different directory structure
    if year <= 2018:
        return [
            f"{BASE}/{year}_PUR_report_textfiles/county_summary_reports"
            f"/_all_counties/counties_by_ai_subtotals_{year}.txt",
        ]
    # 2019-2022 use County_totals/ with a long filename
    long_name = f"{year}_chemical_subtotals_by_county_pounds_applied_ag_apps_acres_treated.txt"
    # 2023 uses a shorter filename
    short_name = f"{year}_chemical_subtotals_by_county_pounds_apps_acres.txt"
    return [
        f"{BASE}/{year}_pur_report_textfiles/County_totals/{long_name}",
        f"{BASE}/{year}_PUR_report_textfiles/County_totals/{long_name}",
        f"{BASE}/{year}_PUR_report_textfiles/County_totals/{short_name}",
        f"{BASE}/{year}_pur_report_textfiles/County_totals/{short_name}",
    ]

# ---------------------------------------------------------------------------
# Chemical → use-type lookup (uppercase CDPR chemical names)
# ---------------------------------------------------------------------------

CHEMICAL_CLASS: dict[str, str] = {
    # FUMIGANTS
    "1,3-DICHLOROPROPENE": "fumigants",
    "CHLOROPICRIN": "fumigants",
    "METHYL BROMIDE": "fumigants",
    "SULFURYL FLUORIDE": "fumigants",
    "POTASSIUM N-METHYLDITHIOCARBAMATE": "fumigants",
    "METAM-SODIUM": "fumigants",
    "METAM-POTASSIUM": "fumigants",
    "DAZOMET": "fumigants",
    "ALLYL ISOTHIOCYANATE": "fumigants",
    "DIMETHYL DISULFIDE": "fumigants",
    "PROPYLENE OXIDE": "fumigants",
    "CARBON BISULFIDE": "fumigants",
    "ETHYLENE OXIDE": "fumigants",
    "METHYL IODIDE": "fumigants",
    "SODIUM METAM": "fumigants",
    # HERBICIDES
    "GLYPHOSATE": "herbicides",
    "GLYPHOSATE, ISOPROPYLAMINE SALT": "herbicides",
    "GLYPHOSATE (IPA SALT)": "herbicides",
    "GLYPHOSATE, POTASSIUM SALT": "herbicides",
    "GLYPHOSATE (K SALT)": "herbicides",
    "GLYPHOSATE, DIMETHYLAMINE SALT": "herbicides",
    "GLYPHOSATE (DMA SALT)": "herbicides",
    "GLYPHOSATE, AMMONIUM SALT": "herbicides",
    "GLYPHOSATE, MONOAMMONIUM SALT": "herbicides",
    "PROPANIL": "herbicides",
    "THIOBENCARB": "herbicides",
    "PARAQUAT DICHLORIDE": "herbicides",
    "DIQUAT DIBROMIDE": "herbicides",
    "OXYFLUORFEN": "herbicides",
    "PENDIMETHALIN": "herbicides",
    "ORYZALIN": "herbicides",
    "FLUAZIFOP-P-BUTYL": "herbicides",
    "SIMAZINE": "herbicides",
    "ATRAZINE": "herbicides",
    "DIURON": "herbicides",
    "TRIFLURALIN": "herbicides",
    "BENSULFURON METHYL": "herbicides",
    "BENSULFURON-METHYL": "herbicides",
    "TRICLOPYR TRIETHYLAMINE SALT": "herbicides",
    "TRICLOPYR BUTOXYETHYL ESTER": "herbicides",
    "TRICLOPYR": "herbicides",
    "CLOPYRALID": "herbicides",
    "2,4-D": "herbicides",
    "2,4-D, DIMETHYLAMINE SALT": "herbicides",
    "2,4-D, ISOPROPYLAMINE SALT": "herbicides",
    "2,4-D, 2-ETHYLHEXYL ESTER": "herbicides",
    "2,4-D DIMETHYLAMINE SALT": "herbicides",
    "MCPA": "herbicides",
    "MCPA, DIMETHYLAMINE SALT": "herbicides",
    "SETHOXYDIM": "herbicides",
    "NAPROPAMIDE": "herbicides",
    "ETHALFLURALIN": "herbicides",
    "CLOMAZONE": "herbicides",
    "HALOSULFURON METHYL": "herbicides",
    "HALOSULFURON-METHYL": "herbicides",
    "IMAZAPYR": "herbicides",
    "IMAZAPYR, ISOPROPYLAMINE SALT": "herbicides",
    "GLUFOSINATE AMMONIUM": "herbicides",
    "GLUFOSINATE-AMMONIUM": "herbicides",
    "GLUFOSINATE": "herbicides",
    "DICHLOBENIL": "herbicides",
    "HEXAZINONE": "herbicides",
    "AMINOPYRALID": "herbicides",
    "PICLORAM": "herbicides",
    "FLURIDONE": "herbicides",
    "ISOXABEN": "herbicides",
    "MESOTRIONE": "herbicides",
    "OXADIAZON": "herbicides",
    "BENFLURALIN": "herbicides",
    "BISPYRIBAC SODIUM": "herbicides",
    "CARFENTRAZONE ETHYL": "herbicides",
    "CARFENTRAZONE-ETHYL": "herbicides",
    "FENOXAPROP-P-ETHYL": "herbicides",
    "FLAZASULFURON": "herbicides",
    "FLUMIOXAZIN": "herbicides",
    "MSMA": "herbicides",
    "PELARGONIC ACID": "herbicides",
    "PENOXSULAM": "herbicides",
    "QUINCLORAC": "herbicides",
    "RIMSULFURON": "herbicides",
    "SAFLUFENACIL": "herbicides",
    "SULFOSULFURON": "herbicides",
    "BROMACIL": "herbicides",
    "TERBACIL": "herbicides",
    "ACROLEIN": "herbicides",
    "TEBUTHIURON": "herbicides",
    "SULFOMETURON METHYL": "herbicides",
    "SULFOMETURON-METHYL": "herbicides",
    "IMAZAPIC": "herbicides",
    "CLETHODIM": "herbicides",
    "TEPRALOXYDIM": "herbicides",
    "PINOXADEN": "herbicides",
    "METAMITRON": "herbicides",
    "PRONAMIDE": "herbicides",
    "FOMESAFEN": "herbicides",
    "LINURON": "herbicides",
    "CHLORIMURON ETHYL": "herbicides",
    "METSULFURON METHYL": "herbicides",
    "METSULFURON-METHYL": "herbicides",
    "PYRAFLUFEN ETHYL": "herbicides",
    "TRIBENURON METHYL": "herbicides",
    "IMAZETHAPYR": "herbicides",
    "LACTOFEN": "herbicides",
    "ACIFLUORFEN, SODIUM SALT": "herbicides",
    "ACIFLUORFEN": "herbicides",
    "AMITROLE": "herbicides",
    "DITHIOPYR": "herbicides",
    "FLUROXYPYR": "herbicides",
    "FLUROXYPYR MEPTYL ESTER": "herbicides",
    "IMAZAMOX": "herbicides",
    "INDAZIFLAM": "herbicides",
    "OXADIARGYL": "herbicides",
    "PYRASULFOTOLE": "herbicides",
    "TOPRAMEZONE": "herbicides",
    "FLUAZIFOP-BUTYL": "herbicides",
    "FENOXAPROP-ETHYL": "herbicides",
    # INSECTICIDES / MITICIDES / NEMATICIDES
    "MINERAL OIL": "insecticides",
    "ABAMECTIN": "insecticides",
    "ABAMECTIN, OTHER RELATED": "insecticides",
    "CHLORPYRIFOS": "insecticides",
    "IMIDACLOPRID": "insecticides",
    "PERMETHRIN": "insecticides",
    "BIFENTHRIN": "insecticides",
    "LAMBDA-CYHALOTHRIN": "insecticides",
    "MALATHION": "insecticides",
    "DIAZINON": "insecticides",
    "ACEPHATE": "insecticides",
    "METHOMYL": "insecticides",
    "KAOLIN": "insecticides",
    "DISODIUM OCTABORATE TETRAHYDRATE": "insecticides",
    "DISODIUM OCTABORATE TETRAHYDRA": "insecticides",
    "METHYLATED SOYBEAN OIL": "insecticides",
    "SPIROTETRAMAT": "insecticides",
    "THIAMETHOXAM": "insecticides",
    "CLOTHIANIDIN": "insecticides",
    "ACETAMIPRID": "insecticides",
    "DIMETHOATE": "insecticides",
    "SPINETORAM": "insecticides",
    "SPINOSAD": "insecticides",
    "ETOXAZOLE": "insecticides",
    "HEXYTHIAZOX": "insecticides",
    "FENPYROXIMATE": "insecticides",
    "SPIROMESIFEN": "insecticides",
    "CLOFENTEZINE": "insecticides",
    "BIFENAZATE": "insecticides",
    "CHLORFENAPYR": "insecticides",
    "CYFLUTHRIN": "insecticides",
    "CYPERMETHRIN": "insecticides",
    "DELTAMETHRIN": "insecticides",
    "ESFENVALERATE": "insecticides",
    "FENVALERATE": "insecticides",
    "FLUPYRADIFURONE": "insecticides",
    "GAMMA-CYHALOTHRIN": "insecticides",
    "INDOXACARB": "insecticides",
    "METHOXYFENOZIDE": "insecticides",
    "NALED": "insecticides",
    "PHOSMET": "insecticides",
    "PYRETHRINS": "insecticides",
    "ROTENONE": "insecticides",
    "TOLFENPYRAD": "insecticides",
    "ZETA-CYPERMETHRIN": "insecticides",
    "PETROLEUM OIL": "insecticides",
    "PARAFFINIC OIL": "insecticides",
    "HORTICULTURAL OIL": "insecticides",
    "NEEM OIL": "insecticides",
    "DINOTEFURAN": "insecticides",
    "SULFOXAFLOR": "insecticides",
    "CYANTRANILIPROLE": "insecticides",
    "CHLORANTRANILIPROLE": "insecticides",
    "FLONICAMID": "insecticides",
    "PYRIFLUQUINAZON": "insecticides",
    "ACEQUINOCYL": "insecticides",
    "CHLORPYRIFOS-METHYL": "insecticides",
    "EMAMECTIN BENZOATE": "insecticides",
    "FENOXYCARB": "insecticides",
    "FENPROPATHRIN": "insecticides",
    "FLUVALINATE": "insecticides",
    "TAU-FLUVALINATE": "insecticides",
    "FORMETANATE HYDROCHLORIDE": "insecticides",
    "METHIDATHION": "insecticides",
    "OXAMYL": "insecticides",
    "PHORATE": "insecticides",
    "PROPARGITE": "insecticides",
    "PYRIPROXYFEN": "insecticides",
    "TEBUFENPYRAD": "insecticides",
    "TETRACHLORVINPHOS": "insecticides",
    "FLUENSULFONE": "insecticides",
    "CADUSAFOS": "insecticides",
    "FOSTHIAZATE": "insecticides",
    "ALPHA-CYPERMETHRIN": "insecticides",
    "BUPROFEZIN": "insecticides",
    "CHLORPYRIFOS": "insecticides",
    "CYFLUMETOFEN": "insecticides",
    "DIAFENTHIURON": "insecticides",
    "DICOFOL": "insecticides",
    "FENAZAQUIN": "insecticides",
    "FENPYROXIMATE": "insecticides",
    "LUFENURON": "insecticides",
    "MILBEMECTIN": "insecticides",
    "NEONICOTINOID": "insecticides",
    "PYMETROZINE": "insecticides",
    "PYRIDABEN": "insecticides",
    "SPIROTETRAMAT": "insecticides",
    "SULFUR DIOXIDE": "fumigants",
    # FUNGICIDES
    "SULFUR": "fungicides",
    "COPPER HYDROXIDE": "fungicides",
    "COPPER SULFATE": "fungicides",
    "COPPER SULFATE PENTAHYDRATE": "fungicides",
    "BASIC COPPER SULFATE": "fungicides",
    "COPPER OXIDE": "fungicides",
    "CUPRIC HYDROXIDE": "fungicides",
    "CUPROUS OXIDE": "fungicides",
    "BASIC COPPER CARBONATE": "fungicides",
    "COPPER ETHANOLAMINE COMPLEXES": "fungicides",
    "COPPER ETHANOLAMINE COMPLEXES,": "fungicides",
    "COPPER OCTANOATE": "fungicides",
    "COPPER AMMONIUM COMPLEX": "fungicides",
    "COPPER AMMONIUM COMPLEXES": "fungicides",
    "LIME-SULFUR": "fungicides",
    "LIME SULFUR": "fungicides",
    "CAPTAN": "fungicides",
    "CHLOROTHALONIL": "fungicides",
    "IPRODIONE": "fungicides",
    "MYCLOBUTANIL": "fungicides",
    "TEBUCONAZOLE": "fungicides",
    "PROPICONAZOLE": "fungicides",
    "AZOXYSTROBIN": "fungicides",
    "CYPRODINIL": "fungicides",
    "FLUDIOXONIL": "fungicides",
    "MANCOZEB": "fungicides",
    "METALAXYL": "fungicides",
    "MEFENOXAM": "fungicides",
    "METALAXYL-M": "fungicides",
    "FOSETYL-AL": "fungicides",
    "FOSETYL-ALUMINUM": "fungicides",
    "FOSETYL ALUMINUM": "fungicides",
    "BOSCALID": "fungicides",
    "TRIFLOXYSTROBIN": "fungicides",
    "PENTHIOPYRAD": "fungicides",
    "FLUOPYRAM": "fungicides",
    "ZIRAM": "fungicides",
    "THIRAM": "fungicides",
    "FENARIMOL": "fungicides",
    "PYRIMETHANIL": "fungicides",
    "CYAZOFAMID": "fungicides",
    "MANDIPROPAMID": "fungicides",
    "POTASSIUM BICARBONATE": "fungicides",
    "HYDROGEN DIOXIDE": "fungicides",
    "HYDROGEN PEROXIDE": "fungicides",
    "DODINE": "fungicides",
    "FENBUCONAZOLE": "fungicides",
    "FLUXAPYROXAD": "fungicides",
    "ISOPYRAZAM": "fungicides",
    "KRESOXIM-METHYL": "fungicides",
    "PROPAMOCARB HYDROCHLORIDE": "fungicides",
    "PROPAMOCARB": "fungicides",
    "PYRACLOSTROBIN": "fungicides",
    "QUINOXYFEN": "fungicides",
    "TETRACONAZOLE": "fungicides",
    "THIOPHANATE METHYL": "fungicides",
    "THIOPHANATE-METHYL": "fungicides",
    "TRIADIMEFON": "fungicides",
    "TRITICONAZOLE": "fungicides",
    "VINCLOZOLIN": "fungicides",
    "FENHEXAMID": "fungicides",
    "CYFLUFENAMID": "fungicides",
    "FLUTRIAFOL": "fungicides",
    "POTASSIUM PHOSPHITE": "fungicides",
    "SODIUM BICARBONATE": "fungicides",
    "PHOSPHORIC ACID": "fungicides",
    "DIFENOCONAZOLE": "fungicides",
    "EPOXICONAZOLE": "fungicides",
    "METRAFENONE": "fungicides",
    "PROQUINAZID": "fungicides",
    "PROTHIOCONAZOLE": "fungicides",
    "SEDAXANE": "fungicides",
    "SPIROXAMINE": "fungicides",
    "TRIFLUMIZOLE": "fungicides",
    "IPCONAZOLE": "fungicides",
    "FLUAZINAM": "fungicides",
    "THIABENDAZOLE": "fungicides",
    "BENOMYL": "fungicides",
    "OXINE-COPPER": "fungicides",
    "COPPER LINOLEATE": "fungicides",
    "COPPER NAPHTHENATE": "fungicides",
    "CYMOXANIL": "fungicides",
    "DIMETHOMORPH": "fungicides",
    "FAMOXADONE": "fungicides",
    "FENAMIDONE": "fungicides",
    "IPROVALICARB": "fungicides",
    "OXATHIAPIPROLIN": "fungicides",
    "PEROXYACETIC ACID": "fungicides",
    "COPPER SULFATE (TRIBASIC)": "fungicides",
    "TRIBASIC COPPER SULFATE": "fungicides",
    "ZIRAM": "fungicides",
    "CAPTAN": "fungicides",
    "DICHLORAN": "fungicides",
    "FENPICLONIL": "fungicides",
    "FLUDIOXONIL": "fungicides",
    "IPRODIONE": "fungicides",
    "MANCOZEB": "fungicides",
    "METIRAM": "fungicides",
    "PROPINEB": "fungicides",
    "THIRAM": "fungicides",
    "ZINEB": "fungicides",
    # BACTERICIDES
    "STREPTOMYCIN": "bactericides",
    "STREPTOMYCIN SULFATE": "bactericides",
    "OXYTETRACYCLINE": "bactericides",
    "OXYTETRACYCLINE HYDROCHLORIDE": "bactericides",
    "KASUGAMYCIN": "bactericides",
    "KASUGAMYCIN HYDROCHLORIDE": "bactericides",
    # PLANT GROWTH REGULATORS
    "GIBBERELLIC ACID": "plant_growth_regulators",
    "GIBBERELLIN A4A7": "plant_growth_regulators",
    "GIBBERELLIN A4 A7": "plant_growth_regulators",
    "GIBBERELLINS A4 AND A7": "plant_growth_regulators",
    "GIBBERELLINS": "plant_growth_regulators",
    "ETHEPHON": "plant_growth_regulators",
    "1-METHYLCYCLOPROPENE": "plant_growth_regulators",
    "NAPHTHALENEACETIC ACID": "plant_growth_regulators",
    "NAPHTHYLACETIC ACID": "plant_growth_regulators",
    "AMINOETHOXYVINYLGLYCINE": "plant_growth_regulators",
    "AMINOETHOXYVINYLGLYCINE HYDROCHLORIDE": "plant_growth_regulators",
    "PROHEXADIONE CALCIUM": "plant_growth_regulators",
    "PROHEXADIONE-CALCIUM": "plant_growth_regulators",
    "TRINEXAPAC-ETHYL": "plant_growth_regulators",
    "S-ABSCISIC ACID": "plant_growth_regulators",
    "MEPIQUAT CHLORIDE": "plant_growth_regulators",
    "6-BENZYLADENINE": "plant_growth_regulators",
    "6-BENZYLAMINOPURINE": "plant_growth_regulators",
    "THIDIAZURON": "plant_growth_regulators",
    "FORCHLORFENURON": "plant_growth_regulators",
    "INDOLE-3-BUTYRIC ACID": "plant_growth_regulators",
    "INDOLE BUTYRIC ACID": "plant_growth_regulators",
    "CLOPROP": "plant_growth_regulators",
    "CYCLANILIDE": "plant_growth_regulators",
    "PACLOBUTRAZOL": "plant_growth_regulators",
    "UNICONAZOLE": "plant_growth_regulators",
    "BENZYLADENINE": "plant_growth_regulators",
    "CHLORMEQUAT CHLORIDE": "plant_growth_regulators",
    "ANCYMIDOL": "plant_growth_regulators",
    "DAMINOZIDE": "plant_growth_regulators",
    "MEFLUIDIDE": "plant_growth_regulators",
    # SOIL AMENDMENTS / FERTILIZER ADJUVANTS (classify as other)
    "CALCIUM HYDROXIDE": "other",
    "AMMONIUM SULFATE": "other",
    "UREA DIHYDROGEN SULFATE": "other",
    "DIETHYLENE GLYCOL": "other",
    "DIMETHYLPOLYSILOXANE": "other",
    # SURFACTANTS / ADJUVANTS (other)
    "LECITHIN": "other",
    "VEGETABLE OIL": "other",
    "METHYLATED FATTY ACIDS FROM CANOLA OIL": "other",
    "OLEIC ACID, METHYL ESTER": "other",
    "OLEIC ACID, ETHYL ESTER": "other",
    "FATTY ACIDS, METHYL ESTERS": "other",
    "FATTY ACIDS, C16-C18 AND C18-UNSATURATED, METHYL ESTERS": "other",
    "CAPRYLIC ACID": "other",
    "ALPHA-PINENE BETA-PINENE COPOLYMER": "other",
    # SANITIZERS / DISINFECTANTS (other)
    "SODIUM HYPOCHLORITE": "other",
    "SODIUM CHLORATE": "other",   # also a herbicide/desiccant
    "CHLORINE": "other",
    "SODIUM BROMIDE": "other",
    "SODIUM CARBONATE PEROXYHYDRATE": "other",
    # PHYSICAL INSECTICIDES
    "BORIC ACID": "insecticides",
    "DIATOMACEOUS EARTH": "insecticides",
    "CRYOLITE": "insecticides",
    "PETROLEUM OIL, UNCLASSIFIED": "insecticides",
    # HERBICIDES (missed variants)
    "S-METOLACHLOR": "herbicides",
    "TRICLOPYR, TRIETHYLAMINE SALT": "herbicides",
    "BENSULIDE": "herbicides",
    # BIOLOGICAL / MICROBIAL (other)
    "BURKHOLDERIA SP STRAIN A396 CELLS AND FERMENTATION MEDIA": "other",
    "BACILLUS AMYLOLIQUEFACIENS STRAIN D747": "other",
    "BACILLUS SUBTILIS": "other",
    "BACILLUS THURINGIENSIS": "insecticides",
    # RODENTICIDES
    "ZINC PHOSPHIDE": "rodenticides",
    "BRODIFACOUM": "rodenticides",
    "BROMADIOLONE": "rodenticides",
    "DIPHACINONE": "rodenticides",
    "CHLOROPHACINONE": "rodenticides",
    "STRYCHNINE": "rodenticides",
    "SODIUM FLUOROACETATE": "rodenticides",
    "WARFARIN": "rodenticides",
    "DIFETHIALONE": "rodenticides",
    "ALPHA-CHLOROHYDRIN": "rodenticides",
    "CHOLECALCIFEROL": "rodenticides",
    "DIFENACOUM": "rodenticides",
    "PINDONE": "rodenticides",
}

# Heuristic fallbacks: substring → class (applied in order, uppercase)
HEURISTIC_RULES: list[tuple[str, str]] = [
    ("GLYPHOSATE", "herbicides"),
    ("CONAZOLE", "fungicides"),
    ("OXYSTROBIN", "fungicides"),
    ("STROB", "fungicides"),
    ("COPPER", "fungicides"),
    ("SULFUR", "fungicides"),
    ("CAPTAN", "fungicides"),
    ("MANCOZ", "fungicides"),
    ("METALAX", "fungicides"),
    ("DICARB", "fungicides"),
    ("THIABEND", "fungicides"),
    ("PYRETHRIN", "insecticides"),
    ("NEONICOT", "insecticides"),
    ("IMIDACL", "insecticides"),
    ("THIAMETH", "insecticides"),
    ("CLOTHIAN", "insecticides"),
    ("ACETAMIP", "insecticides"),
    ("SPINOSAD", "insecticides"),
    ("ABAMECT", "insecticides"),
    ("CHLORPYRI", "insecticides"),
    ("CYPERM", "insecticides"),
    ("FLUAZ", "herbicides"),
    ("PROPAN", "herbicides"),
    ("URON", "herbicides"),
    ("TRIFLU", "herbicides"),
    ("AMIDE", "herbicides"),
    ("GIBBEREL", "plant_growth_regulators"),
    ("ETHEPHON", "plant_growth_regulators"),
    ("MEPIQUAT", "plant_growth_regulators"),
    ("RODENTICIDE", "rodenticides"),
    ("ANTICOAG", "rodenticides"),
    ("STREPTOM", "bactericides"),
    ("TETRACYCL", "bactericides"),
]

ALL_CLASSES = [
    "fumigants",
    "herbicides",
    "insecticides",
    "fungicides",
    "bactericides",
    "plant_growth_regulators",
    "rodenticides",
    "other",
]

# CA county land area in square miles (Census Bureau)
COUNTY_AREA: dict[str, float] = {
    "Alameda": 738.0, "Alpine": 738.4, "Amador": 594.7, "Butte": 1636.5,
    "Calaveras": 1020.4, "Colusa": 1150.6, "Contra Costa": 719.7,
    "Del Norte": 1007.9, "El Dorado": 1712.0, "Fresno": 5962.9,
    "Glenn": 1314.9, "Humboldt": 3573.1, "Imperial": 4175.0,
    "Inyo": 10226.6, "Kern": 8140.7, "Kings": 1389.5, "Lake": 1257.5,
    "Lassen": 4557.4, "Los Angeles": 4057.9, "Madera": 2136.5,
    "Marin": 519.7, "Mariposa": 1451.2, "Mendocino": 3506.6,
    "Merced": 1929.3, "Modoc": 3943.9, "Mono": 3043.8,
    "Monterey": 3321.4, "Napa": 753.8, "Nevada": 974.3,
    "Orange": 789.7, "Placer": 1502.7, "Plumas": 2553.6,
    "Riverside": 7206.5, "Sacramento": 994.0, "San Benito": 1389.2,
    "San Bernardino": 20105.5, "San Diego": 4261.0, "San Francisco": 46.9,
    "San Joaquin": 1399.4, "San Luis Obispo": 3299.2, "San Mateo": 449.0,
    "Santa Barbara": 2737.5, "Santa Clara": 1291.0, "Santa Cruz": 445.5,
    "Shasta": 3784.9, "Sierra": 953.3, "Siskiyou": 6346.6,
    "Solano": 827.9, "Sonoma": 1575.6, "Stanislaus": 1494.9,
    "Sutter": 607.8, "Tehama": 2950.5, "Trinity": 3179.3,
    "Tulare": 4863.3, "Tuolumne": 2235.1, "Ventura": 1845.1,
    "Yolo": 1023.6, "Yuba": 638.9,
}


def classify(chemical: object) -> str:
    if not isinstance(chemical, str) or not chemical.strip():
        return "other"
    key = chemical.upper().strip()
    if key in CHEMICAL_CLASS:
        return CHEMICAL_CLASS[key]
    for fragment, cls in HEURISTIC_RULES:
        if fragment in key:
            return cls
    return "other"


def normalize_county(raw: str) -> str:
    """Convert CDPR all-caps county name to title-case."""
    return raw.strip().title()


def fetch_year(year: int) -> pd.DataFrame | None:
    for url in county_subtotals_url(year):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                print(f"  {year}: fetched {len(r.content):,} bytes")
                text = r.text

                # Detect delimiter: CDPR files are tab-separated
                lines = [l for l in text.splitlines() if l.strip()]
                if not lines:
                    continue

                # Try to detect if there is a header row
                first = lines[0].split("\t")
                has_header = not first[0].strip().isdigit()

                df = pd.read_csv(
                    io.StringIO(text),
                    sep="\t",
                    header=0 if has_header else None,
                    names=None if has_header else [
                        "year", "county", "chemical", "lbs_applied",
                        "applications", "area_treated",
                    ],
                    dtype=str,
                )
                df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

                # Rename columns to canonical names if they differ
                rename_map = {}
                for col in df.columns:
                    if "pound" in col or col in ("lbs_applied", "lbs", "pounds"):
                        rename_map[col] = "lbs_applied"
                    elif "county" in col:
                        rename_map[col] = "county"
                    elif "chem" in col:
                        rename_map[col] = "chemical"
                    elif "year" in col:
                        rename_map[col] = "year"
                df = df.rename(columns=rename_map)

                required = {"county", "chemical", "lbs_applied"}
                if not required.issubset(df.columns):
                    print(f"    WARNING: unexpected columns {list(df.columns)}, skipping")
                    continue

                # Handle CDPR "<0.01" sentinel values — treat as 0
                df["lbs_applied"] = (
                    df["lbs_applied"]
                    .str.replace(r"^<.*", "0", regex=True)
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0)
                )
                df["year"] = year
                df["county"] = df["county"].apply(normalize_county)
                df["chemical"] = df["chemical"].str.strip()
                return df[["year", "county", "chemical", "lbs_applied"]]
        except Exception as e:
            print(f"    {url}: {e}")
    print(f"  {year}: no data found")
    return None


def build_ts(all_data: list[pd.DataFrame]) -> str:
    df = pd.concat(all_data, ignore_index=True)
    df["use_type"] = df["chemical"].apply(classify)

    years_present = sorted(df["year"].unique())
    num_years = len(years_present)

    # Track unclassified chemicals by total lbs for diagnostics
    unclassified = (
        df[df["use_type"] == "other"]
        .groupby("chemical")["lbs_applied"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
    )
    if not unclassified.empty:
        print("\nTop unclassified chemicals (lbs across all counties/years):")
        for chem, lbs in unclassified.items():
            print(f"  {chem}: {lbs:,.0f} lbs")

    # Aggregate: county × year × use_type → total lbs
    agg = (
        df.groupby(["county", "year", "use_type"])["lbs_applied"]
        .sum()
        .reset_index()
    )

    # Top ingredients per county (across all years, by total lbs)
    top_ing = (
        df.groupby(["county", "chemical", "use_type"])["lbs_applied"]
        .sum()
        .reset_index()
        .sort_values("lbs_applied", ascending=False)
        .groupby("county")
        .head(5)
    )

    counties = sorted(df["county"].unique())
    output_counties: list[dict] = []

    for county in counties:
        area = COUNTY_AREA.get(county)
        if area is None:
            print(f"  WARNING: no area for county '{county}', skipping")
            continue

        county_agg = agg[agg["county"] == county]

        # per-year breakdown
        by_year: dict[int, dict] = {}
        for year in years_present:
            yr_data = county_agg[county_agg["year"] == year]
            by_class: dict[str, float] = {cls: 0.0 for cls in ALL_CLASSES}
            for _, row in yr_data.iterrows():
                by_class[row["use_type"]] = round(row["lbs_applied"] / area)
            total = round(sum(by_class.values()))
            by_year[int(year)] = {
                "total": total,
                "by_class": {k: int(v) for k, v in by_class.items()},
            }

        # overall averages
        avg_by_class: dict[str, float] = {}
        for cls in ALL_CLASSES:
            vals = [by_year[y]["by_class"][cls] for y in years_present]
            avg_by_class[cls] = round(sum(vals) / num_years)
        overall_avg = round(sum(avg_by_class.values()))

        # total lbs applied across all years
        total_lbs = round(
            df[(df["county"] == county)]["lbs_applied"].sum()
        )

        # top pesticide class by average lbs/sq mi
        top_class = max(avg_by_class, key=lambda k: avg_by_class[k])  # type: ignore

        # top 5 ingredients
        county_ing = top_ing[top_ing["county"] == county].head(5)
        top_ingredients = [
            {
                "name": row["chemical"].title(),
                "lbs_applied": int(row["lbs_applied"]),
                "category": row["use_type"],
            }
            for _, row in county_ing.iterrows()
        ]

        output_counties.append({
            "county": county,
            "lbs_per_sq_mile": overall_avg,
            "lbs_applied_total": total_lbs,
            "top_pesticide_class": top_class,
            "by_class": {k: int(v) for k, v in avg_by_class.items()},
            "top_ingredients": top_ingredients,
            "by_year": by_year,
        })

    # Sort descending by lbs_per_sq_mile
    output_counties.sort(key=lambda c: c["lbs_per_sq_mile"], reverse=True)

    min_year = min(years_present)
    max_year = max(years_present)

    lines: list[str] = [
        "// California pesticide use data by county.",
        "// Source: CDPR Pesticide Use Reporting (PUR) database.",
        "// https://files.cdpr.ca.gov/pub/outgoing/pur/data/",
        f"// Values: lbs of active ingredient per square mile ({min_year}–{max_year} annual).",
        "// Generated by database/seed/generate_pesticide_data.py — do not edit by hand.",
        "",
        "export type PesticideClass =",
        "  | 'fumigants'",
        "  | 'herbicides'",
        "  | 'insecticides'",
        "  | 'fungicides'",
        "  | 'bactericides'",
        "  | 'plant_growth_regulators'",
        "  | 'rodenticides'",
        "  | 'other';",
        "",
        "export const PESTICIDE_CLASSES: { value: PesticideClass; label: string }[] = [",
        "  { value: 'fumigants', label: 'Fumigants' },",
        "  { value: 'herbicides', label: 'Herbicides' },",
        "  { value: 'insecticides', label: 'Insecticides' },",
        "  { value: 'fungicides', label: 'Fungicides' },",
        "  { value: 'bactericides', label: 'Bactericides' },",
        "  { value: 'plant_growth_regulators', label: 'Plant Growth Regulators' },",
        "  { value: 'rodenticides', label: 'Rodenticides' },",
        "  { value: 'other', label: 'Other' },",
        "];",
        "",
        "export interface ActiveIngredient {",
        "  name: string;",
        "  lbs_applied: number;",
        "  category: PesticideClass;",
        "}",
        "",
        "export interface YearData {",
        "  total: number;",
        "  by_class: Record<PesticideClass, number>;",
        "}",
        "",
        "export interface CountyPesticideData {",
        "  county: string;",
        "  lbs_per_sq_mile: number;",
        "  lbs_applied_total: number;",
        "  top_pesticide_class: PesticideClass;",
        "  by_class: Record<PesticideClass, number>;",
        "  top_ingredients: ActiveIngredient[];",
        "  by_year: Record<number, YearData>;",
        "}",
        "",
    ]

    # Emit data
    lines.append(f"export const PESTICIDE_DATA: CountyPesticideData[] = [")
    for c in output_counties:
        lines.append("  {")
        lines.append(f"    county: {json.dumps(c['county'])},")
        lines.append(f"    lbs_per_sq_mile: {c['lbs_per_sq_mile']},")
        lines.append(f"    lbs_applied_total: {c['lbs_applied_total']},")
        lines.append(f"    top_pesticide_class: '{c['top_pesticide_class']}',")
        # by_class
        bc = c["by_class"]
        bc_parts = ", ".join(f"{k}: {bc[k]}" for k in ALL_CLASSES)
        lines.append(f"    by_class: {{ {bc_parts} }},")
        # top_ingredients
        if c["top_ingredients"]:
            lines.append("    top_ingredients: [")
            for ing in c["top_ingredients"]:
                lines.append(
                    f"      {{ name: {json.dumps(ing['name'])}, "
                    f"lbs_applied: {ing['lbs_applied']}, "
                    f"category: '{ing['category']}' }},"
                )
            lines.append("    ],")
        else:
            lines.append("    top_ingredients: [],")
        # by_year
        lines.append("    by_year: {")
        for yr in sorted(c["by_year"].keys()):
            yd = c["by_year"][yr]
            bc2 = yd["by_class"]
            bc2_parts = ", ".join(f"{k}: {bc2[k]}" for k in ALL_CLASSES)
            lines.append(
                f"      {yr}: {{ total: {yd['total']}, "
                f"by_class: {{ {bc2_parts} }} }},"
            )
        lines.append("    },")
        lines.append("  },")
    lines.append("];")
    lines.append("")
    lines.append("export const PESTICIDE_BY_COUNTY: Record<string, CountyPesticideData> =")
    lines.append("  Object.fromEntries(PESTICIDE_DATA.map(d => [d.county, d]));")
    lines.append("")

    # ── PESTICIDE_BY_CHEMICAL ────────────────────────────────────────────────
    # chemical (title-case) → county → avg lbs/sq mi across all years
    chem_county = (
        df.groupby(["county", "chemical"])["lbs_applied"]
        .sum()
        .reset_index()
    )
    chem_county["chemical_tc"] = chem_county["chemical"].apply(
        lambda c: c.strip().title() if isinstance(c, str) else ""
    )
    chem_county = chem_county[chem_county["chemical_tc"] != ""]

    by_chemical: dict[str, dict[str, int]] = {}
    for _, row in chem_county.iterrows():
        county = row["county"]
        chem = row["chemical_tc"]
        area = COUNTY_AREA.get(county)
        if area is None:
            continue
        val = round(row["lbs_applied"] / area / num_years)
        if val == 0:
            continue
        if chem not in by_chemical:
            by_chemical[chem] = {}
        by_chemical[chem][county] = val

    # sort chemicals alphabetically for stable output
    sorted_chemicals = sorted(by_chemical.keys())

    lines.append("// Pesticides featured in the Tracking California Pesticide Mapping Tool")
    lines.append("export const TRACKING_CA_PESTICIDES: readonly string[] = [")
    tracking_ca = [
        "Sulfur",
        "Mineral Oil",
        "1,3-Dichloropropene",
        "Potassium N-Methyldithiocarbamate",
        "Petroleum Oil, Unclassified",
        "Chloropicrin",
        "Glyphosate, Potassium Salt",
        "Glyphosate, Isopropylamine Salt",
        "Metam-Sodium",
        "Kaolin",
        "Copper Hydroxide",
        "Pendimethalin",
    ]
    for name in tracking_ca:
        lines.append(f"  {json.dumps(name)},")
    lines.append("] as const;")
    lines.append("")

    lines.append("// chemical → county → avg lbs/sq mi (2016–2023)")
    lines.append("export const PESTICIDE_BY_CHEMICAL: Record<string, Record<string, number>> = {")
    for chem in sorted_chemicals:
        county_vals = by_chemical[chem]
        inner = ", ".join(
            f"{json.dumps(c)}: {v}"
            for c, v in sorted(county_vals.items())
        )
        lines.append(f"  {json.dumps(chem)}: {{ {inner} }},")
    lines.append("};")
    lines.append("")

    print(f"  {len(sorted_chemicals)} unique chemicals indexed")
    return "\n".join(lines)


def main() -> None:
    print("Fetching CDPR county chemical subtotals...")
    frames: list[pd.DataFrame] = []
    for year in YEARS:
        df = fetch_year(year)
        if df is not None:
            frames.append(df)

    if not frames:
        print("ERROR: no data fetched")
        sys.exit(1)

    print(f"\nBuilding TypeScript ({len(frames)} years, "
          f"{sum(len(f) for f in frames):,} records)...")
    ts = build_ts(frames)

    out = Path(__file__).parent.parent.parent / "frontend" / "src" / "data" / "pesticideData.ts"
    out.write_text(ts)
    print(f"\nWrote {len(ts):,} bytes to {out}")
    print("Done.")


if __name__ == "__main__":
    main()
