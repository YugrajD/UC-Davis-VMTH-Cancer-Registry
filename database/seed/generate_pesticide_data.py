#!/usr/bin/env python3
"""
Generate frontend/src/data/pesticideData.ts from CDPR PUR county-level data.

Downloads chemical-subtotals-by-county files for 2016-2023, maps each chemical
against the Tracking California Pesticide Mapping Tool's health-effect categories
(Carcinogens, Cholinesterase Inhibitors, Endocrine Disruptors, Fumigants,
Neonicotinoids, Reproductive & Developmental Toxicants, Toxic Air Contaminants),
and outputs a TypeScript module with all 58 CA counties and per-year, per-category
lbs/sq-mi data.  A chemical may belong to multiple categories; lbs are counted
toward every applicable category (overlapping totals).

Category source: Tracking California Pesticide Mapping Tool chemical categories
spreadsheet, last updated June 2019.
https://trackingcalifornia.org/images/uploads/pesticide-mapping-tool-chemical-categories.xlsx

Usage:
    python3 database/seed/generate_pesticide_data.py
"""

import io
import json
import sys
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
# Tracking California PPHC chemical → health-effect categories
# Source: pesticide-mapping-tool-chemical-categories.xlsx (June 2019)
# A chemical may belong to multiple categories.
# ---------------------------------------------------------------------------

PPHC_CATEGORIES: dict[str, frozenset[str]] = {
    '1,1,2-TRICHLOROETHANE': frozenset({'toxic_air_contaminants'}),
    '1,1-DIMETHYL HYDRAZINE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    '1,2,3-PROPANE TRICARBOXYLIC ACID, 2-HYDROXY, TRILITHIUM SALT': frozenset({'repro_developmental'}),
    '1,2,4-TRICHLOROBENZENE': frozenset({'toxic_air_contaminants'}),
    '1,2-DICHLOROPROPANE': frozenset({'fumigants', 'toxic_air_contaminants'}),
    '1,2-DICHLOROPROPANE, 1,3-DICHLOROPROPENE AND RELATED C3 COMPOUNDS': frozenset({'carcinogens', 'fumigants'}),
    '1,2-EPOXY BUTANE': frozenset({'toxic_air_contaminants'}),
    '1,3-BUTADIENE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    '1,3-DICHLOROPROPENE': frozenset({'carcinogens', 'fumigants', 'toxic_air_contaminants'}),
    "10,10'-OXYBISPHENOXYARSINE": frozenset({'carcinogens'}),
    '1080': frozenset({'repro_developmental'}),
    '2,4,5-TRICHLOROPHENOL': frozenset({'toxic_air_contaminants'}),
    '2,4,6-TRICHLOROPHENOL': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    '2,4,6-TRICHLOROPHENOL, SODIUM SALT': frozenset({'carcinogens'}),
    '2,4-D': frozenset({'toxic_air_contaminants'}),
    '2,4-D, 2-ETHYLHEXYL ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, ALKANOLAMINE SALTS (ETHANOL AND ISOPROPANOL AMINES)': frozenset({'toxic_air_contaminants'}),
    '2,4-D, BUTOXY ETHOXY PROPANOL ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, BUTOXYETHANOL ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, BUTOXYPROPYL ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, BUTYL ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, DIETHANOLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, DIETHYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, DIMETHYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, DODECYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, HEPTYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, ISOOCTYL ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, ISOPROPYL ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, ISOPROPYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, LITHIUM SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, MORPHOLINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, N,N-DIMETHYL OLEYL-LINOLEYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, N-OLEYL-1,3-PROPYLENEDIAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, OCTYL ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, PROPYL ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, PROPYLENE GLYCOL BUTYL ETHER ESTER': frozenset({'toxic_air_contaminants'}),
    '2,4-D, SODIUM SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, TETRADECYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, TRIETHANOLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, TRIETHYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, TRIISOPROPANOLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-D, TRIISOPROPYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2,4-DB ACID': frozenset({'toxic_air_contaminants'}),
    '2,4-DINITROPHENOL': frozenset({'toxic_air_contaminants'}),
    '2-(2,4-DP), DIMETHYLAMINE SALT': frozenset({'toxic_air_contaminants'}),
    '2-BUTOXYETHANOL': frozenset({'toxic_air_contaminants'}),
    '2-NITROPROPANE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    '3,5,5-TRIMETHYL-2-CYCLOHEXEN-1-ONE': frozenset({'toxic_air_contaminants'}),
    '3-CHLORO-2-METHYL PROPENE': frozenset({'carcinogens'}),
    '3-HYDROXYCARBOFURAN': frozenset({'cholinesterase_inhibitors'}),
    '3-IODO-2-PROPYNYL BUTYLCARBAMATE': frozenset({'cholinesterase_inhibitors'}),
    '4-CPA': frozenset({'carcinogens'}),
    '4-CPA, DIETHANOLAMINE SALT': frozenset({'carcinogens'}),
    'ABAMECTIN': frozenset({'repro_developmental'}),
    'ACEPHATE': frozenset({'cholinesterase_inhibitors'}),
    'ACETALDEHYDE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'ACETAMIPRID': frozenset({'neonicotinoids'}),
    'ACETOCHLOR': frozenset({'carcinogens', 'endocrine_disruptors'}),
    'ACETONITRILE': frozenset({'toxic_air_contaminants'}),
    'ACETOPHENONE': frozenset({'toxic_air_contaminants'}),
    'ACIFLUORFEN, SODIUM SALT': frozenset({'carcinogens'}),
    'ACROLEIN': frozenset({'toxic_air_contaminants'}),
    'ACRYLIC ACID': frozenset({'toxic_air_contaminants'}),
    'ACRYLONITRILE': frozenset({'carcinogens', 'fumigants', 'toxic_air_contaminants'}),
    'AKTON': frozenset({'cholinesterase_inhibitors'}),
    'AKTON, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'ALACHLOR': frozenset({'carcinogens', 'endocrine_disruptors'}),
    'ALDICARB': frozenset({'cholinesterase_inhibitors'}),
    'ALDICARB SULFOXIDE': frozenset({'cholinesterase_inhibitors'}),
    'ALDRIN': frozenset({'carcinogens'}),
    'ALUMINUM PHOSPHIDE': frozenset({'fumigants', 'toxic_air_contaminants'}),
    'AMINOCARB': frozenset({'cholinesterase_inhibitors'}),
    'AMITRAZ': frozenset({'repro_developmental'}),
    'AMITROLE': frozenset({'carcinogens'}),
    'AMMONIUM ARSENATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'ANILINE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'ANTIMONY': frozenset({'toxic_air_contaminants'}),
    'ANTIMONY POTASSIUM TARTRATE': frozenset({'toxic_air_contaminants'}),
    'ARAMITE': frozenset({'carcinogens'}),
    'ARSENIC': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'ARSENIC ACID': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'ARSENIC PENTOXIDE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'ARSENIC TRIOXIDE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'ASBESTOS': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'ATRAZINE': frozenset({'endocrine_disruptors', 'repro_developmental'}),
    'AURAMINE': frozenset({'carcinogens'}),
    'AZINPHOS-ETHYL': frozenset({'cholinesterase_inhibitors'}),
    'AZINPHOS-METHYL': frozenset({'cholinesterase_inhibitors'}),
    'AZINPHOS-METHYL OXYGEN ANALOG': frozenset({'cholinesterase_inhibitors'}),
    'B-MYRCENE': frozenset({'carcinogens'}),
    'BENDIOCARB': frozenset({'cholinesterase_inhibitors'}),
    'BENOMYL': frozenset({'repro_developmental'}),
    'BENSULIDE': frozenset({'cholinesterase_inhibitors'}),
    'BENZENE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'BENZYL CHLORIDE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'BETA-BUTYROLACTONE': frozenset({'carcinogens'}),
    'BHA': frozenset({'carcinogens'}),
    'BHC (OTHER THAN GAMMA ISOMER)': frozenset({'carcinogens'}),
    'BIPHENYL': frozenset({'toxic_air_contaminants'}),
    'BIS-(2,2-DICHLOROETHYL) ETHER': frozenset({'toxic_air_contaminants'}),
    'BIS-(CHLOROETHYL) ETHER': frozenset({'carcinogens'}),
    'BISPHENOL A': frozenset({'endocrine_disruptors'}),
    'BROMACIL, LITHIUM SALT': frozenset({'repro_developmental'}),
    'BROMOETHANE': frozenset({'carcinogens'}),
    'BROMOXYNIL BUTYRATE': frozenset({'repro_developmental'}),
    'BROMOXYNIL HEPTANOATE': frozenset({'repro_developmental'}),
    'BROMOXYNIL OCTANOATE': frozenset({'repro_developmental'}),
    'BROMOXYNIL PHENOL': frozenset({'repro_developmental'}),
    'BUFENCARB': frozenset({'cholinesterase_inhibitors'}),
    'BUTACHLOR': frozenset({'carcinogens'}),
    'BUTATHIOFOS': frozenset({'cholinesterase_inhibitors'}),
    'BUTOXYCARBOXIM': frozenset({'cholinesterase_inhibitors'}),
    'BUTYL BENZYL PHTHALATE': frozenset({'endocrine_disruptors', 'repro_developmental'}),
    'BUTYLATE': frozenset({'cholinesterase_inhibitors'}),
    'CACODYLIC ACID': frozenset({'carcinogens'}),
    'CADMIUM CARBONATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CADMIUM CHLORIDE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CADMIUM COCOATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CADMIUM COMPOUNDS': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CADMIUM PERBORATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CADMIUM SEBACATE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'CADMIUM SUCCINATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CADMIUM YELLOW PIGMENTS': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CALCIUM ACID METHANEARSONATE': frozenset({'carcinogens'}),
    'CALCIUM ARSENATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CALCIUM CYANAMIDE': frozenset({'toxic_air_contaminants'}),
    'CALCIUM CYANIDE': frozenset({'toxic_air_contaminants'}),
    'CAPTAFOL': frozenset({'carcinogens', 'cholinesterase_inhibitors'}),
    'CAPTAN': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CAPTAN, OTHER RELATED': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CARBARYL': frozenset({'carcinogens', 'cholinesterase_inhibitors', 'repro_developmental', 'toxic_air_contaminants'}),
    'CARBOFURAN': frozenset({'cholinesterase_inhibitors'}),
    'CARBON BLACK PIGMENT': frozenset({'carcinogens'}),
    'CARBON DIOXIDE': frozenset({'fumigants'}),
    'CARBON DISULFIDE': frozenset({'fumigants', 'repro_developmental', 'toxic_air_contaminants'}),
    'CARBON TETRACHLORIDE': frozenset({'carcinogens', 'fumigants', 'toxic_air_contaminants'}),
    'CARBOPHENOTHION': frozenset({'cholinesterase_inhibitors'}),
    'CARBOSULFAN': frozenset({'cholinesterase_inhibitors'}),
    'CHLORACETIC ACID': frozenset({'toxic_air_contaminants'}),
    'CHLORAMBEN': frozenset({'toxic_air_contaminants'}),
    'CHLORDANE': frozenset({'carcinogens', 'endocrine_disruptors', 'toxic_air_contaminants'}),
    'CHLORDECONE': frozenset({'carcinogens', 'endocrine_disruptors', 'repro_developmental'}),
    'CHLORDIMEFORM': frozenset({'carcinogens'}),
    'CHLORDIMEFORM HYDROCHLORIDE': frozenset({'carcinogens'}),
    'CHLORETHOXYPHOS': frozenset({'cholinesterase_inhibitors'}),
    'CHLORFENVINPHOS': frozenset({'cholinesterase_inhibitors'}),
    'CHLORINE': frozenset({'toxic_air_contaminants'}),
    'CHLORO DIFLUORO METHANE': frozenset({'fumigants'}),
    'CHLOROBENZENE': frozenset({'toxic_air_contaminants'}),
    'CHLOROBENZILATE': frozenset({'toxic_air_contaminants'}),
    'CHLOROFORM': frozenset({'carcinogens', 'fumigants', 'repro_developmental', 'toxic_air_contaminants'}),
    'CHLOROMETHOXY PROPYL MERCURIC ACETAMIDE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'CHLOROPICRIN': frozenset({'fumigants', 'toxic_air_contaminants'}),
    'CHLOROTETRACYCLINE': frozenset({'repro_developmental'}),
    'CHLOROTHALONIL': frozenset({'carcinogens'}),
    'CHLORPROPHAM': frozenset({'cholinesterase_inhibitors'}),
    'CHLORPYRIFOS': frozenset({'cholinesterase_inhibitors', 'repro_developmental'}),
    'CHLORPYRIFOS OXON': frozenset({'cholinesterase_inhibitors'}),
    'CHLORPYRIFOS-METHYL': frozenset({'cholinesterase_inhibitors'}),
    'CHLORTHIOPHOS': frozenset({'cholinesterase_inhibitors'}),
    'CHROMIC ACID': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'CHROMIUM DIOXIDE': frozenset({'toxic_air_contaminants'}),
    'CLOTHIANIDIN': frozenset({'neonicotinoids'}),
    'COBALT NAPHTHENATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'COBALT OCTOATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'COBALTOUS SULFATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'COPPER-ZINC CHROMATE COMPLEX': frozenset({'carcinogens'}),
    'COUMAPHOS': frozenset({'cholinesterase_inhibitors'}),
    'COUMAPHOS, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'CREOSOTE': frozenset({'carcinogens'}),
    'CROTOXYPHOS': frozenset({'cholinesterase_inhibitors'}),
    'CROTOXYPHOS, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'CUMENE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'CUPROUS THIOCYANATE': frozenset({'toxic_air_contaminants'}),
    'CYANAZINE': frozenset({'repro_developmental'}),
    'CYCLOATE': frozenset({'cholinesterase_inhibitors', 'repro_developmental'}),
    'CYCLOHEXIMIDE': frozenset({'repro_developmental'}),
    'CYHEXATIN': frozenset({'repro_developmental'}),
    'CYPROCONAZOLE': frozenset({'carcinogens'}),
    'DAMINOZIDE': frozenset({'carcinogens'}),
    'DAZOMET': frozenset({'fumigants', 'toxic_air_contaminants'}),
    'DAZOMET, SODIUM SALT': frozenset({'fumigants'}),
    'DBCP': frozenset({'carcinogens', 'fumigants', 'repro_developmental', 'toxic_air_contaminants'}),
    'DBCP, OTHER RELATED': frozenset({'fumigants'}),
    'DDD': frozenset({'carcinogens'}),
    'DDE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'DDT': frozenset({'carcinogens', 'endocrine_disruptors', 'repro_developmental'}),
    'DDVP': frozenset({'carcinogens', 'cholinesterase_inhibitors', 'toxic_air_contaminants'}),
    'DDVP, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'DEMETON': frozenset({'cholinesterase_inhibitors'}),
    'DEMETON-S-METHYL': frozenset({'cholinesterase_inhibitors'}),
    'DESMEDIPHAM': frozenset({'cholinesterase_inhibitors'}),
    'DIALIFOR': frozenset({'cholinesterase_inhibitors'}),
    'DIALIFOR, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'DIALLATE': frozenset({'cholinesterase_inhibitors'}),
    'DIAZINON': frozenset({'cholinesterase_inhibitors'}),
    'DIAZOXON': frozenset({'cholinesterase_inhibitors'}),
    'DIBUTYLPHTHALATE': frozenset({'endocrine_disruptors', 'repro_developmental', 'toxic_air_contaminants'}),
    'DICHLOFENTHION': frozenset({'cholinesterase_inhibitors'}),
    'DICHLORMATE': frozenset({'cholinesterase_inhibitors'}),
    'DICHLORO ACETIC ACID': frozenset({'carcinogens'}),
    'DICLOFOP-METHYL': frozenset({'carcinogens', 'repro_developmental'}),
    'DICROTOPHOS': frozenset({'cholinesterase_inhibitors'}),
    'DIELDRIN': frozenset({'carcinogens'}),
    'DIETHANOLAMINE': frozenset({'toxic_air_contaminants'}),
    'DIETHYLENE GLYCOL MONOETHYL ETHER': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'DIETHYLENE GLYCOL MONOMETHYL ETHER': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'DIMETHOATE': frozenset({'cholinesterase_inhibitors'}),
    'DINOCAP': frozenset({'repro_developmental'}),
    'DINOSEB': frozenset({'repro_developmental'}),
    'DINOTEFURAN': frozenset({'neonicotinoids'}),
    'DIOCTYL PHTHALATE': frozenset({'carcinogens', 'endocrine_disruptors', 'repro_developmental', 'toxic_air_contaminants'}),
    'DIOXACARB': frozenset({'cholinesterase_inhibitors'}),
    'DIOXANE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'DIOXATHION': frozenset({'cholinesterase_inhibitors'}),
    'DIOXATHION, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'DIOXIN': frozenset({'endocrine_disruptors', 'repro_developmental', 'toxic_air_contaminants'}),
    'DIPROPYL ISOCINCHOMERONATE': frozenset({'carcinogens'}),
    'DISODIUM CYANODITHIOIMIDO CARBONATE': frozenset({'repro_developmental'}),
    'DISULFOTON': frozenset({'cholinesterase_inhibitors'}),
    'DIURON': frozenset({'carcinogens'}),
    'DNOC, SODIUM SALT': frozenset({'toxic_air_contaminants'}),
    'DODECYL AMMONIUM METHANEARSONATE': frozenset({'carcinogens'}),
    'DOXORUBICIN': frozenset({'carcinogens', 'repro_developmental'}),
    'DSMA': frozenset({'carcinogens'}),
    'EDIFENPHOS': frozenset({'cholinesterase_inhibitors'}),
    'EDTA, DISODIUM MANGANESE SALT': frozenset({'toxic_air_contaminants'}),
    'EHTYL FORMATE': frozenset({'fumigants'}),
    'ENDOSULFAN': frozenset({'toxic_air_contaminants'}),
    'ENDRIN': frozenset({'repro_developmental'}),
    'EPICHLOROHYDRIN': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'EPN': frozenset({'cholinesterase_inhibitors'}),
    'EPTC': frozenset({'cholinesterase_inhibitors', 'repro_developmental'}),
    'ERBON': frozenset({'carcinogens'}),
    'ESBIOTHRIN': frozenset({'carcinogens'}),
    'ETHEPHON': frozenset({'cholinesterase_inhibitors'}),
    'ETHION': frozenset({'cholinesterase_inhibitors'}),
    'ETHOPROP': frozenset({'carcinogens', 'cholinesterase_inhibitors'}),
    'ETHYL ACRYLATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'ETHYL BENZENE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'ETHYL CHLORIDE': frozenset({'toxic_air_contaminants'}),
    'ETHYLENE DIBROMIDE': frozenset({'carcinogens', 'fumigants', 'repro_developmental', 'toxic_air_contaminants'}),
    'ETHYLENE DICHLORIDE': frozenset({'carcinogens', 'fumigants', 'toxic_air_contaminants'}),
    'ETHYLENE GLYCOL': frozenset({'toxic_air_contaminants'}),
    'ETHYLENE GLYCOL MONOMETHYL ETHER': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'ETHYLENE OXIDE': frozenset({'carcinogens', 'fumigants', 'repro_developmental', 'toxic_air_contaminants'}),
    'ETHYLENE THIOUREA': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'ETHYLMERCURIC PHOSPHATE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'FAMPHUR': frozenset({'cholinesterase_inhibitors'}),
    'FENAMIPHOS': frozenset({'cholinesterase_inhibitors'}),
    'FENAMIPHOS SULFONE': frozenset({'cholinesterase_inhibitors'}),
    'FENAMIPHOS SULFOXIDE': frozenset({'cholinesterase_inhibitors'}),
    'FENOXAPROP-ETHYL': frozenset({'repro_developmental'}),
    'FENOXYCARB': frozenset({'carcinogens'}),
    'FENSULFOTHION': frozenset({'cholinesterase_inhibitors'}),
    'FENTHION': frozenset({'cholinesterase_inhibitors'}),
    'FENTIN HYDROXIDE': frozenset({'carcinogens', 'endocrine_disruptors'}),
    'FLUAZIFOP-BUTYL': frozenset({'repro_developmental'}),
    'FLUAZIFOP-P-BUTYL': frozenset({'repro_developmental'}),
    'FOLPET': frozenset({'carcinogens'}),
    'FONOFOS': frozenset({'cholinesterase_inhibitors'}),
    'FORMALDEHYDE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'FORMETANATE HYDROCHLORIDE': frozenset({'cholinesterase_inhibitors'}),
    'FOSPIRATE': frozenset({'cholinesterase_inhibitors'}),
    'FREON 12': frozenset({'fumigants'}),
    'GLYPHOSATE': frozenset({'carcinogens'}),
    'GLYPHOSATE DIAMMONIUM SALT': frozenset({'carcinogens'}),
    'GLYPHOSATE DIMETHYLAMINE SALT': frozenset({'carcinogens'}),
    'GLYPHOSATE ISOPROPYLAMINE SALT': frozenset({'carcinogens'}),
    'GLYPHOSATE MONOAMMONIUM SALT': frozenset({'carcinogens'}),
    'GLYPHOSATE POTASSIUM SALT': frozenset({'carcinogens'}),
    'GLYPHOSATE-TRIMESIUM': frozenset({'carcinogens'}),
    'HEPTACHLOR': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'HEPTACHLOR EPOXIDE': frozenset({'carcinogens'}),
    'HEXACHLOROBENZENE': frozenset({'carcinogens', 'endocrine_disruptors', 'repro_developmental', 'toxic_air_contaminants'}),
    'HEXYTHIAZOX': frozenset({'carcinogens'}),
    'HYDRAMETHYLNON': frozenset({'repro_developmental'}),
    'HYDRAZINE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'HYDROGEN CHLORIDE': frozenset({'fumigants', 'toxic_air_contaminants'}),
    'HYDROQUINONE': frozenset({'toxic_air_contaminants'}),
    'HYDROXY MERCURI NITROPHENOL': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'IMAZALIL': frozenset({'carcinogens'}),
    'IMIDACLOPRID': frozenset({'neonicotinoids'}),
    'IODOMETHANE': frozenset({'carcinogens', 'fumigants', 'toxic_air_contaminants'}),
    'IPRODIONE': frozenset({'carcinogens'}),
    'ISAZOPHOS': frozenset({'cholinesterase_inhibitors'}),
    'ISOFENPHOS': frozenset({'cholinesterase_inhibitors'}),
    'KRESOXIM-METHYL': frozenset({'carcinogens'}),
    'LACTOFEN': frozenset({'carcinogens'}),
    'LAURIC ACID, BARIUM CADMIUM SALT': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'LEAD': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'LEAD ARSENATE (STANDARD)': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'LEAD ARSENATE, BASIC': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'LEAD METASILICATE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'LEAD MONOXIDE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'LEPTOPHOS': frozenset({'cholinesterase_inhibitors'}),
    'LEPTOPHOS, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'LIGNIN SULFONIC ACID, MANGANESE SALT': frozenset({'toxic_air_contaminants'}),
    'LIGNIN SULFONIC ACID, ZINC, MANGANESE & IRON SALTS': frozenset({'toxic_air_contaminants'}),
    'LINDANE': frozenset({'carcinogens', 'endocrine_disruptors', 'toxic_air_contaminants'}),
    'LINURON': frozenset({'endocrine_disruptors', 'repro_developmental'}),
    'LITHIUM CARBONATE': frozenset({'repro_developmental'}),
    'MAGNESIUM PHOSPHIDE': frozenset({'fumigants', 'toxic_air_contaminants'}),
    'MAGNESIUM SILICATE': frozenset({'carcinogens'}),
    'MALAOXON': frozenset({'cholinesterase_inhibitors'}),
    'MALATHION': frozenset({'carcinogens', 'cholinesterase_inhibitors'}),
    'MALEIC ANHYDRIDE': frozenset({'toxic_air_contaminants'}),
    'MANCOZEB': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'MANEB': frozenset({'carcinogens', 'endocrine_disruptors', 'toxic_air_contaminants'}),
    'MANGANESE (II) OXIDE': frozenset({'toxic_air_contaminants'}),
    'MANGANESE CARBAMATE': frozenset({'toxic_air_contaminants'}),
    'MANGANESE SULFATE': frozenset({'toxic_air_contaminants'}),
    'MERCURIC CHLORIDE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'MERCURIC DIMETHYL DITHIOCARBAMATE': frozenset({'cholinesterase_inhibitors', 'repro_developmental', 'toxic_air_contaminants'}),
    'MERCURIC OLEATE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'MERCURIC OXIDE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'MERPHOS': frozenset({'cholinesterase_inhibitors'}),
    'MERPHOS, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'META-CRESOL': frozenset({'toxic_air_contaminants'}),
    'METAM-SODIUM': frozenset({'carcinogens', 'cholinesterase_inhibitors', 'endocrine_disruptors', 'fumigants', 'repro_developmental', 'toxic_air_contaminants'}),
    'METHAMIDOPHOS': frozenset({'cholinesterase_inhibitors'}),
    'METHANOL': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'METHAZOLE': frozenset({'repro_developmental'}),
    'METHIDATHION': frozenset({'cholinesterase_inhibitors', 'toxic_air_contaminants'}),
    'METHIDATHION OXON': frozenset({'cholinesterase_inhibitors'}),
    'METHIOCARB': frozenset({'cholinesterase_inhibitors'}),
    'METHIOCARB SULFONE': frozenset({'cholinesterase_inhibitors'}),
    'METHIOCARB SULFOXIDE': frozenset({'cholinesterase_inhibitors'}),
    'METHOMYL': frozenset({'cholinesterase_inhibitors'}),
    'METHOXYCHLOR': frozenset({'toxic_air_contaminants'}),
    'METHOXYCHLOR, OTHER RELATED': frozenset({'toxic_air_contaminants'}),
    'METHYL BROMIDE': frozenset({'fumigants', 'repro_developmental', 'toxic_air_contaminants'}),
    'METHYL CHLORIDE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'METHYL ETHYL KETONE': frozenset({'toxic_air_contaminants'}),
    'METHYL FORMATE': frozenset({'fumigants'}),
    'METHYL ISOBUTYL KETONE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'METHYL ISOTHIOCYANATE': frozenset({'carcinogens', 'fumigants', 'toxic_air_contaminants'}),
    'METHYL METHACRYLATE': frozenset({'toxic_air_contaminants'}),
    'METHYL PARAOXON': frozenset({'cholinesterase_inhibitors'}),
    'METHYL PARATHION': frozenset({'cholinesterase_inhibitors', 'toxic_air_contaminants'}),
    'METHYL PARATHION, OTHER RELATED': frozenset({'cholinesterase_inhibitors', 'toxic_air_contaminants'}),
    'METHYL TRITHION': frozenset({'cholinesterase_inhibitors'}),
    'METHYL-CARBOFENTHION': frozenset({'cholinesterase_inhibitors'}),
    'METHYLENE CHLORIDE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'METHYLMERCURY DICYANO DIAMIDE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'METIRAM': frozenset({'carcinogens', 'repro_developmental'}),
    'METRONIDAZOLE': frozenset({'carcinogens'}),
    'MEVINPHOS': frozenset({'cholinesterase_inhibitors'}),
    'MEVINPHOS, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'MEXACARBATE': frozenset({'cholinesterase_inhibitors'}),
    'MIREX': frozenset({'carcinogens', 'endocrine_disruptors'}),
    'MOLINATE': frozenset({'cholinesterase_inhibitors', 'repro_developmental'}),
    'MOLINATE SULFOXIDE': frozenset({'cholinesterase_inhibitors', 'repro_developmental'}),
    'MONOCROTOPHOS': frozenset({'cholinesterase_inhibitors'}),
    'MSMA': frozenset({'carcinogens'}),
    'MYCLOBUTANIL': frozenset({'repro_developmental'}),
    'N,N-DIMETHYLFORMAMIDE': frozenset({'toxic_air_contaminants'}),
    'N-2-FLUORENYL ACETAMIDE': frozenset({'toxic_air_contaminants'}),
    'N-HEXANE': frozenset({'toxic_air_contaminants'}),
    'NABAM': frozenset({'repro_developmental'}),
    'NALED': frozenset({'cholinesterase_inhibitors', 'toxic_air_contaminants'}),
    'NAPHTHA, HEAVY AROMATIC': frozenset({'carcinogens'}),
    'NAPHTHALENE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'NAPHTHENIC ACID, LEAD SALT': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'NEOMYCIN SULFATE': frozenset({'repro_developmental'}),
    'NICKEL': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'NICKEL DIETHYL HEXYL ACID PHOSPHATE COMPLEX': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'NICOTINE': frozenset({'repro_developmental'}),
    'NITRAPYRIN': frozenset({'carcinogens', 'repro_developmental'}),
    'NITROFEN': frozenset({'carcinogens'}),
    'NITROMETHANE': frozenset({'carcinogens'}),
    'NITROUS OXIDE': frozenset({'repro_developmental'}),
    'NOURY DRY COBALT 12%': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'NUODEX NAPHTHENATE COBALT 6% CATALYST': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'NUXTRA MANGANESE 12% CATALYST': frozenset({'toxic_air_contaminants'}),
    'NUXTRA MANGANESE 9% CATALYST': frozenset({'toxic_air_contaminants'}),
    'O,O-DIETHYL PHOSPHORO CHLORIDOTHIONATE': frozenset({'cholinesterase_inhibitors'}),
    'O,O-DIETHYL-O-PHENYL PHOSPHOROTHIOATE': frozenset({'cholinesterase_inhibitors'}),
    'O,O-DIMETHYL O-(4-NITRO-M-TOLYL) PHOSPHOROTHIOATE': frozenset({'cholinesterase_inhibitors'}),
    'O-CRESOL': frozenset({'toxic_air_contaminants'}),
    'OCHRATOXIN A': frozenset({'carcinogens'}),
    'OCTYLAMMONIUM METHANEARSONATE': frozenset({'carcinogens'}),
    'OMETHOATE': frozenset({'cholinesterase_inhibitors'}),
    'ORTHO-PHENYLPHENOL': frozenset({'carcinogens'}),
    'OXADIAZON': frozenset({'carcinogens', 'repro_developmental'}),
    'OXAMYL': frozenset({'cholinesterase_inhibitors'}),
    'OXYDEMETON-METHYL': frozenset({'cholinesterase_inhibitors', 'repro_developmental'}),
    'OXYTETRACYCLINE HYDROCHLORIDE': frozenset({'repro_developmental'}),
    'OXYTETRACYCLINE, CALCIUM COMPLEX': frozenset({'repro_developmental'}),
    'OXYTHIOQUINOX': frozenset({'carcinogens', 'fumigants', 'repro_developmental'}),
    'P-CRESOL': frozenset({'toxic_air_contaminants'}),
    'P-NITROPHENOL': frozenset({'toxic_air_contaminants'}),
    'PARA-DICHLOROBENZENE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'PARAOXON': frozenset({'cholinesterase_inhibitors'}),
    'PARATHION': frozenset({'cholinesterase_inhibitors', 'toxic_air_contaminants'}),
    'PARATHION, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'PARIS GREEN': frozenset({'carcinogens'}),
    'PCNB': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'PCP, OTHER RELATED': frozenset({'carcinogens'}),
    'PCP, POTASSIUM SALT': frozenset({'carcinogens'}),
    'PCP, SODIUM SALT': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'PCP, SODIUM SALT, OTHER RELATED': frozenset({'carcinogens'}),
    'PEBULATE': frozenset({'cholinesterase_inhibitors'}),
    'PENTACHLOROPHENOL': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'PHENARSAZINE CHLORIDE': frozenset({'carcinogens'}),
    'PHENMEDIPHAM': frozenset({'cholinesterase_inhibitors'}),
    'PHENOL': frozenset({'toxic_air_contaminants'}),
    'PHENYL GLYCOL ETHER': frozenset({'toxic_air_contaminants'}),
    'PHENYLMERCURIC ACETATE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'PHENYLMERCURIC ACETATE, OTHER RELATED': frozenset({'toxic_air_contaminants'}),
    'PHENYLMERCURIC AMMONIUM ACETATE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'PHENYLMERCURIC LACTATE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'PHENYLMERCURIC NITRATE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'PHENYLMERCURIC OLEATE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'PHORATE': frozenset({'cholinesterase_inhibitors'}),
    'PHORATE SULFONE': frozenset({'cholinesterase_inhibitors'}),
    'PHORATE SULFOXIDE': frozenset({'cholinesterase_inhibitors'}),
    'PHORATOXON': frozenset({'cholinesterase_inhibitors'}),
    'PHORATOXON SULFONE': frozenset({'cholinesterase_inhibitors'}),
    'PHORATOXON SULFOXIDE': frozenset({'cholinesterase_inhibitors'}),
    'PHOSACETIN': frozenset({'cholinesterase_inhibitors'}),
    'PHOSALONE': frozenset({'cholinesterase_inhibitors'}),
    'PHOSALONE OXON': frozenset({'cholinesterase_inhibitors'}),
    'PHOSMET': frozenset({'cholinesterase_inhibitors'}),
    'PHOSMETOXON': frozenset({'cholinesterase_inhibitors'}),
    'PHOSPHAMIDON': frozenset({'cholinesterase_inhibitors'}),
    'PHOSPHAMIDON, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'PHOSPHINE': frozenset({'fumigants', 'toxic_air_contaminants'}),
    'PHOSPHORUS': frozenset({'toxic_air_contaminants'}),
    'PHOSTEBUPIRIM': frozenset({'cholinesterase_inhibitors'}),
    'PHTHALIC ANHYDRIDE': frozenset({'toxic_air_contaminants'}),
    'PIRIMICARB': frozenset({'cholinesterase_inhibitors'}),
    'PIRIMIPHOS ETHYL': frozenset({'cholinesterase_inhibitors'}),
    'PIRIMIPHOS-METHYL': frozenset({'cholinesterase_inhibitors'}),
    'POLY ALKYLENE GLYCOL ETHERS': frozenset({'toxic_air_contaminants'}),
    'POLYCHLOROBIPHENYL': frozenset({'carcinogens', 'endocrine_disruptors', 'toxic_air_contaminants'}),
    'POTASSIUM CHROMATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'POTASSIUM DICHROMATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'POTASSIUM DIMETHYL DITHIO CARBAMATE': frozenset({'cholinesterase_inhibitors', 'repro_developmental'}),
    'POTASSIUM N-METHYLDITHIOCARBAMATE': frozenset({'carcinogens', 'fumigants', 'repro_developmental', 'toxic_air_contaminants'}),
    'POTASSIUM PERMANGANATE': frozenset({'toxic_air_contaminants'}),
    'PROCYMIDONE': frozenset({'carcinogens'}),
    'PROFENOFOS': frozenset({'cholinesterase_inhibitors'}),
    'PROMECARB': frozenset({'cholinesterase_inhibitors'}),
    'PROPACHLOR': frozenset({'carcinogens'}),
    'PROPAMOCARB HYDROCHLORIDE': frozenset({'cholinesterase_inhibitors'}),
    'PROPARGITE': frozenset({'carcinogens', 'repro_developmental'}),
    'PROPAZINE': frozenset({'repro_developmental'}),
    'PROPETAMPHOS': frozenset({'cholinesterase_inhibitors'}),
    'PROPOXUR': frozenset({'carcinogens', 'cholinesterase_inhibitors', 'toxic_air_contaminants'}),
    'PROPOXUR, OTHER RELATED': frozenset({'carcinogens', 'cholinesterase_inhibitors'}),
    'PROPYLENE OXIDE': frozenset({'carcinogens', 'fumigants', 'toxic_air_contaminants'}),
    'PROTHIOFOS': frozenset({'cholinesterase_inhibitors'}),
    'PYMETROZINE': frozenset({'carcinogens'}),
    'PYRAFLUFEN-ETHYL': frozenset({'carcinogens'}),
    'PYRIDINE': frozenset({'carcinogens'}),
    'QUIZALOFOP-ETHYL': frozenset({'repro_developmental'}),
    'RESMETHRIN': frozenset({'carcinogens', 'repro_developmental'}),
    'RESORCINOL': frozenset({'endocrine_disruptors'}),
    'RONNEL': frozenset({'cholinesterase_inhibitors'}),
    'S,S,S-TRIBUTYL PHOSPHOROTRITHIOATE': frozenset({'carcinogens', 'cholinesterase_inhibitors', 'toxic_air_contaminants'}),
    'SAFROLE': frozenset({'carcinogens'}),
    'SEDAXANE': frozenset({'carcinogens'}),
    'SILICA AEROGEL': frozenset({'carcinogens'}),
    'SILICA GEL': frozenset({'carcinogens'}),
    'SILICA, CRYSTALLINE-QUARTZ': frozenset({'carcinogens'}),
    'SILVEX': frozenset({'carcinogens'}),
    'SILVEX, PROPYLENE GLYCOL BUTYL ETHER ESTER': frozenset({'carcinogens'}),
    'SILVEX, TRIETHANOLAMINE SALT': frozenset({'carcinogens'}),
    'SIMAZINE': frozenset({'repro_developmental'}),
    'SODIUM ARSENATE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'SODIUM ARSENITE': frozenset({'carcinogens', 'repro_developmental', 'toxic_air_contaminants'}),
    'SODIUM BICHROMATE DIHYDRATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'SODIUM CACODYLATE': frozenset({'carcinogens'}),
    'SODIUM CHROMATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'SODIUM CYANIDE': frozenset({'toxic_air_contaminants'}),
    'SODIUM DICHROMATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'SODIUM DIMETHYL DITHIO CARBAMATE': frozenset({'cholinesterase_inhibitors', 'repro_developmental'}),
    'SODIUM PYROARSENATE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'SODIUM TETRATHIOCARBONATE': frozenset({'fumigants', 'toxic_air_contaminants'}),
    'SPIRODICLOFEN': frozenset({'carcinogens'}),
    'STREPTOMYCIN': frozenset({'repro_developmental'}),
    'STREPTOMYCIN SULFATE': frozenset({'repro_developmental'}),
    'SULFALLATE': frozenset({'carcinogens'}),
    'SULFOTEP': frozenset({'cholinesterase_inhibitors'}),
    'SULFOTEP, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'SULFUR DIOXIDE': frozenset({'repro_developmental'}),
    'SULFURYL FLUORIDE': frozenset({'fumigants', 'toxic_air_contaminants'}),
    'SULPROFOS': frozenset({'cholinesterase_inhibitors'}),
    'TALC': frozenset({'carcinogens'}),
    'TAU-FLUVALINATE': frozenset({'repro_developmental'}),
    'TEMEPHOS': frozenset({'cholinesterase_inhibitors'}),
    'TEPP': frozenset({'cholinesterase_inhibitors'}),
    'TEPP, OTHER RELATED': frozenset({'cholinesterase_inhibitors'}),
    'TERBACIL': frozenset({'repro_developmental'}),
    'TERBUFOS': frozenset({'cholinesterase_inhibitors'}),
    'TERBUTOL': frozenset({'cholinesterase_inhibitors'}),
    'TERRAZOLE': frozenset({'carcinogens'}),
    'TETRACHLOROETHANE': frozenset({'carcinogens', 'fumigants', 'toxic_air_contaminants'}),
    'TETRACHLOROETHYLENE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'TETRACHLORVINPHOS': frozenset({'carcinogens', 'cholinesterase_inhibitors'}),
    'TETRACONAZOLE': frozenset({'carcinogens'}),
    'THIABENDAZOLE': frozenset({'carcinogens'}),
    'THIAMETHOXAM': frozenset({'neonicotinoids'}),
    'THIOBENCARB': frozenset({'cholinesterase_inhibitors'}),
    'THIOBENCARB SULFOXIDE': frozenset({'cholinesterase_inhibitors'}),
    'THIODICARB': frozenset({'carcinogens', 'cholinesterase_inhibitors'}),
    'THIOFANOX': frozenset({'cholinesterase_inhibitors'}),
    'THIONAZIN': frozenset({'cholinesterase_inhibitors'}),
    'THIOPHANATE-METHYL': frozenset({'carcinogens', 'repro_developmental'}),
    'THIOUREA': frozenset({'carcinogens'}),
    'THIRAM': frozenset({'endocrine_disruptors'}),
    'TOLUENE': frozenset({'repro_developmental', 'toxic_air_contaminants'}),
    'TOLUENE, 2,4-DIISOCYANATE': frozenset({'carcinogens'}),
    'TOLYLFLUANID': frozenset({'carcinogens'}),
    'TOXAPHENE': frozenset({'carcinogens', 'endocrine_disruptors', 'toxic_air_contaminants'}),
    'TRIADIMEFON': frozenset({'repro_developmental'}),
    'TRIAZOPHOS': frozenset({'cholinesterase_inhibitors'}),
    'TRIBUTYLTIN BENZOATE': frozenset({'endocrine_disruptors'}),
    'TRIBUTYLTIN FLUORIDE': frozenset({'endocrine_disruptors'}),
    'TRIBUTYLTIN LINOLEATE': frozenset({'endocrine_disruptors'}),
    'TRIBUTYLTIN METHACRYLATE': frozenset({'endocrine_disruptors', 'repro_developmental'}),
    'TRIBUTYLTIN OXIDE': frozenset({'endocrine_disruptors'}),
    'TRICHLORFON': frozenset({'cholinesterase_inhibitors'}),
    'TRICHLORO ETHYLENE': frozenset({'carcinogens', 'toxic_air_contaminants'}),
    'TRICHLOROFLUOROMETHANE': frozenset({'fumigants'}),
    'TRICHLORONATE': frozenset({'cholinesterase_inhibitors'}),
    'TRIFLURALIN': frozenset({'toxic_air_contaminants'}),
    'TRIFORINE': frozenset({'repro_developmental'}),
    'TRIMETHACARB': frozenset({'cholinesterase_inhibitors'}),
    'VAMIDOTHION': frozenset({'cholinesterase_inhibitors'}),
    'VINCLOZOLIN': frozenset({'carcinogens', 'endocrine_disruptors', 'repro_developmental'}),
    'VINYL ACETATE': frozenset({'toxic_air_contaminants'}),
    'WARFARIN': frozenset({'repro_developmental'}),
    'WARFARIN, SODIUM SALT': frozenset({'repro_developmental'}),
    'XYLENE': frozenset({'toxic_air_contaminants'}),
    'ZINC PHOSPHIDE': frozenset({'fumigants', 'toxic_air_contaminants'}),
    'ZINEB': frozenset({'endocrine_disruptors'}),
}

ALL_CLASSES = [
    "carcinogens",
    "cholinesterase_inhibitors",
    "endocrine_disruptors",
    "fumigants",
    "neonicotinoids",
    "repro_developmental",
    "toxic_air_contaminants",
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


def classify(chemical: object) -> list[str]:
    """Return list of PPHC category keys for a chemical (may be empty or multi-valued)."""
    if not isinstance(chemical, str) or not chemical.strip():
        return []
    key = chemical.upper().strip()
    return list(PPHC_CATEGORIES.get(key, frozenset()))


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

    years_present = sorted(df["year"].unique())
    num_years = len(years_present)

    # Assign PPHC categories (multi-valued list per row)
    df["categories"] = df["chemical"].apply(classify)

    # Explode: one row per (county, year, chemical, category) for PPHC chemicals only
    df_exp = df.explode("categories").rename(columns={"categories": "use_type"})
    df_exp = df_exp[df_exp["use_type"].notna() & (df_exp["use_type"] != "")]

    # Count of PPHC-categorized vs total chemicals
    pphc_chemical_count = df[df["categories"].apply(len) > 0]["chemical"].nunique()
    total_chemical_count = df["chemical"].nunique()
    print(f"  {pphc_chemical_count} / {total_chemical_count} unique chemicals matched to a PPHC category")

    # Aggregate PPHC categories: county × year × use_type → total lbs
    agg = (
        df_exp.groupby(["county", "year", "use_type"])["lbs_applied"]
        .sum()
        .reset_index()
    )

    # Total lbs per county × year (ALL chemicals — used for .total and lbs_per_sq_mile)
    total_by_county_year = (
        df.groupby(["county", "year"])["lbs_applied"]
        .sum()
        .reset_index()
    )

    # Top ingredients per county (all chemicals, by total lbs across all years)
    top_ing = (
        df.groupby(["county", "chemical"])["lbs_applied"]
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
            print(f"  WARNING: no area for county '{{county}}', skipping")
            continue

        county_agg = agg[agg["county"] == county]
        county_totals = total_by_county_year[total_by_county_year["county"] == county]

        # per-year breakdown
        by_year: dict[int, dict] = {}
        for year in years_present:
            yr_data = county_agg[county_agg["year"] == year]
            yr_total_rows = county_totals[county_totals["year"] == year]

            by_class: dict[str, float] = {cls: 0.0 for cls in ALL_CLASSES}
            for _, row in yr_data.iterrows():
                by_class[row["use_type"]] = round(row["lbs_applied"] / area)

            # Use actual total (not sum of PPHC categories, which overlap)
            raw_total = yr_total_rows["lbs_applied"].sum() if not yr_total_rows.empty else 0
            total = round(raw_total / area)

            by_year[int(year)] = {
                "total": total,
                "by_class": {k: int(v) for k, v in by_class.items()},
            }

        # overall averages per category
        avg_by_class: dict[str, float] = {}
        for cls in ALL_CLASSES:
            vals = [by_year[y]["by_class"][cls] for y in years_present]
            avg_by_class[cls] = round(sum(vals) / num_years)

        # lbs_per_sq_mile = average of annual totals across all chemicals
        year_totals = [by_year[y]["total"] for y in years_present]
        overall_avg = round(sum(year_totals) / num_years)

        # total lbs applied across all years
        total_lbs = round(df[df["county"] == county]["lbs_applied"].sum())

        # top PPHC category by average lbs/sq mi
        top_class = max(avg_by_class, key=lambda k: avg_by_class[k])  # type: ignore

        # top 5 ingredients (all chemicals)
        county_ing = top_ing[top_ing["county"] == county].head(5)
        top_ingredients = [
            {
                "name": row["chemical"].title(),
                "lbs_applied": int(row["lbs_applied"]),
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
        f"// Values: lbs of active ingredient per square mile ({min_year}\u2013{max_year} annual).",
        "// Health-effect categories from Tracking California Pesticide Mapping Tool (June 2019).",
        "// A chemical may appear in multiple categories; category totals overlap.",
        "// Generated by database/seed/generate_pesticide_data.py \u2014 do not edit by hand.",
        "",
        "export type PesticideClass =",
        "  | 'carcinogens'",
        "  | 'cholinesterase_inhibitors'",
        "  | 'endocrine_disruptors'",
        "  | 'fumigants'",
        "  | 'neonicotinoids'",
        "  | 'repro_developmental'",
        "  | 'toxic_air_contaminants';",
        "",
        "export const PESTICIDE_CLASSES: { value: PesticideClass; label: string }[] = [",
        "  { value: 'carcinogens', label: 'Carcinogens' },",
        "  { value: 'cholinesterase_inhibitors', label: 'Cholinesterase Inhibitors' },",
        "  { value: 'endocrine_disruptors', label: 'Endocrine Disruptors' },",
        "  { value: 'fumigants', label: 'Fumigants' },",
        "  { value: 'neonicotinoids', label: 'Neonicotinoids' },",
        "  { value: 'repro_developmental', label: 'Reproductive & Developmental Toxicants' },",
        "  { value: 'toxic_air_contaminants', label: 'Toxic Air Contaminants' },",
        "];",
        "",
        "export interface ActiveIngredient {",
        "  name: string;",
        "  lbs_applied: number;",
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
    lines.append("export const PESTICIDE_DATA: CountyPesticideData[] = [")
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
                    f"lbs_applied: {ing['lbs_applied']} }},"
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

    # \u2500\u2500 PESTICIDE_BY_CHEMICAL \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # chemical (title-case) \u2192 county \u2192 avg lbs/sq mi across all years
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

    lines.append("// chemical \u2192 county \u2192 avg lbs/sq mi (2016\u20132023)")
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
