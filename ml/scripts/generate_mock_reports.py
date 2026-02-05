#!/usr/bin/env python3
"""
Generate mock veterinary pathology reports for the VMTH Cancer Registry.
This script can be run standalone to generate sample reports for testing.
"""

import random
import json


REPORT_TEMPLATES = {
    "Lymphoma": [
        "Histopathologic examination of the {site} reveals a diffuse infiltrate of neoplastic round cells consistent with {subtype} lymphoma. The neoplastic lymphocytes are large with prominent nucleoli and frequent mitotic figures ({mitotic}/10 HPF). Immunohistochemistry confirms {marker} phenotype. Diagnosis: {grade} grade lymphoma. Prognosis is {prognosis} with appropriate chemotherapy.",
        "Biopsy of the {site} demonstrates effacement of normal architecture by a monomorphic population of neoplastic lymphoid cells. Flow cytometry reveals {marker} immunophenotype. Grade: {grade}. The patient presents with stage {stage} multicentric lymphoma. Recommended staging workup includes thoracic radiographs and abdominal ultrasound.",
    ],
    "Mast Cell Tumor": [
        "Excisional biopsy of the {site} mass reveals a dermal neoplasm composed of round cells with metachromatic cytoplasmic granules consistent with mast cells. Patnaik grade {patnaik}. Mitotic index: {mitotic}/10 HPF. Surgical margins are {margins}. Ki-67 index: {ki67}%. c-KIT immunohistochemistry shows {ckit} pattern. Recommend {followup}.",
        "Histopathology of the cutaneous mass from the {site} demonstrates sheets of well-granulated round cells. Kiupel grade: {kiupel}. Toluidine blue staining confirms metachromatic granules. Lateral margins: {margins}. Deep margin: {margins}. Ki-67 proliferation index: {ki67}%.",
    ],
    "Osteosarcoma": [
        "Biopsy of the {bone} reveals a highly cellular neoplasm producing osteoid matrix. Neoplastic osteoblasts display marked anisocytosis and anisokaryosis with {mitotic} mitotic figures per 10 HPF. Areas of necrosis comprising approximately {necrosis}% of the sample. {subtype} variant. Alkaline phosphatase is strongly positive. Diagnosis: High-grade osteosarcoma. Staging recommended prior to limb amputation.",
    ],
    "Hemangiosarcoma": [
        "Splenectomy specimen contains a {size} cm hemorrhagic mass. Histopathology demonstrates a malignant vascular neoplasm with irregular anastomosing channels lined by pleomorphic endothelial cells. Mitotic index: {mitotic}/10 HPF. Immunohistochemistry: CD31 positive, vWF positive, cytokeratin negative. DIC panel recommended. Diagnosis: Splenic hemangiosarcoma, stage {stage}.",
    ],
    "Melanoma": [
        "Biopsy of the {site} mass reveals a malignant melanocytic neoplasm. Tumor cells contain variable melanin pigment. Nuclear atypia is {atypia} with {mitotic} mitotic figures per 10 HPF. Melan-A positive, PNL2 positive, S-100 positive. Margins: {margins}. Melanoma vaccine (Oncept) may be considered. Diagnosis: Oral malignant melanoma.",
    ],
    "Squamous Cell Carcinoma": [
        "Biopsy of the {site} reveals invasive nests and cords of neoplastic squamous epithelial cells. Keratin pearl formation is {keratin}. Desmoplastic stromal response noted. Mitotic figures: {mitotic}/10 HPF. {differentiation} differentiated squamous cell carcinoma. Solar elastosis present in adjacent dermis suggesting UV-induced etiology.",
    ],
    "Fibrosarcoma": [
        "Histopathologic examination of the {site} mass reveals a malignant mesenchymal neoplasm composed of interlacing bundles of spindle cells with collagen production. Herringbone pattern focally present. Mitotic index: {mitotic}/10 HPF. Vimentin positive, S-100 negative, desmin negative. Margins: {margins}. Diagnosis: Fibrosarcoma, grade {grade}.",
    ],
    "Transitional Cell Carcinoma": [
        "Cystoscopic biopsy of the urinary bladder reveals a papillary neoplasm composed of multilayered transitional epithelium with cytologic atypia. {invasion} invasion into lamina propria. Mitotic figures: {mitotic}/10 HPF. Uroplakin III positive. BRAF mutation testing recommended. Diagnosis: Transitional cell carcinoma, grade {grade}. Piroxicam and/or mitoxantrone therapy may be considered.",
    ],
}

FILLS = {
    "site": ["submandibular lymph node", "prescapular lymph node", "mesenteric lymph node",
             "popliteal lymph node", "oral cavity", "nasal cavity", "skin", "digit",
             "ear pinna", "ventral abdomen", "inguinal region", "perianal region", "gingiva",
             "left forelimb", "right hindlimb", "thoracic wall"],
    "subtype": ["B-cell", "T-cell", "large cell", "small cell", "immunoblastic"],
    "marker": ["CD20+ B-cell", "CD3+ T-cell", "CD79a+ B-cell"],
    "grade": ["low", "intermediate", "high"],
    "mitotic": ["2", "5", "8", "12", "18", "25", "32"],
    "prognosis": ["guarded", "fair", "poor"],
    "stage": ["I", "II", "III", "IV"],
    "patnaik": ["I", "II", "III"],
    "kiupel": ["low", "high"],
    "margins": ["clean (>3mm)", "narrow (<1mm)", "incomplete/dirty", "clean (>5mm)"],
    "ki67": ["5", "12", "18", "25", "35"],
    "ckit": ["perimembranous (pattern I)", "focal cytoplasmic (pattern II)",
             "diffuse cytoplasmic (pattern III)"],
    "followup": ["monitoring", "adjuvant radiation therapy",
                 "re-excision with wider margins", "vinblastine/prednisone protocol"],
    "bone": ["distal radius", "proximal humerus", "distal femur", "proximal tibia"],
    "necrosis": ["10", "20", "30", "50"],
    "size": ["3.5", "5.2", "7.8", "10.3"],
    "atypia": ["mild", "moderate", "marked"],
    "keratin": ["prominent", "moderate", "minimal"],
    "differentiation": ["Well", "Moderately", "Poorly"],
    "invasion": ["Superficial", "Deep", "No"],
}


def generate_report(cancer_type: str) -> str:
    templates = REPORT_TEMPLATES.get(cancer_type, REPORT_TEMPLATES["Lymphoma"])
    template = random.choice(templates)
    result = template
    for key, values in FILLS.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


if __name__ == "__main__":
    random.seed(42)
    cancer_types = list(REPORT_TEMPLATES.keys())
    reports = []

    for i in range(50):
        ct = random.choice(cancer_types)
        report = generate_report(ct)
        reports.append({
            "cancer_type": ct,
            "report_text": report,
        })
        print(f"\n--- {ct} ---")
        print(report[:200] + "...")

    # Save as JSON for testing
    with open("/tmp/mock_reports.json", "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\n\nGenerated {len(reports)} mock reports -> /tmp/mock_reports.json")
