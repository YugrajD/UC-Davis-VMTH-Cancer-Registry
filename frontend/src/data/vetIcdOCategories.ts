// Organ-system categories from the VET-ICD-O-Canine-1 classification
// (Vet-ICD-O-canine-1, https://pmc.ncbi.nlm.nih.gov/articles/PMC8946502/).
// Cancer types in this project come from free-text PetBERT predictions, so we
// classify by substring match on the common/default primary site in dogs.

export type VetIcdOCategoryId =
  | 'bone'
  | 'soft_tissue'
  | 'skin'
  | 'genital'
  | 'nervous'
  | 'respiratory'
  | 'hematopoietic'
  | 'ocular_otic'
  | 'alimentary'
  | 'urinary'
  | 'mammary'
  | 'endocrine'
  | 'other';

export interface VetIcdOCategory {
  id: VetIcdOCategoryId;
  label: string;
}

export const VET_ICD_O_CATEGORIES: VetIcdOCategory[] = [
  { id: 'bone', label: 'Bone & Hard Tissues' },
  { id: 'soft_tissue', label: 'Soft Tissue' },
  { id: 'skin', label: 'Skin & Melanocytic' },
  { id: 'genital', label: 'Genital Tract' },
  { id: 'nervous', label: 'Nervous System' },
  { id: 'respiratory', label: 'Respiratory' },
  { id: 'hematopoietic', label: 'Hematopoietic' },
  { id: 'ocular_otic', label: 'Ocular & Otic' },
  { id: 'alimentary', label: 'Alimentary' },
  { id: 'urinary', label: 'Urinary Tract' },
  { id: 'mammary', label: 'Mammary Gland' },
  { id: 'endocrine', label: 'Endocrine' },
  { id: 'other', label: 'Other / Unspecified' },
];

// Keyword rules: the first rule whose keyword is a substring of the cancer
// type name (case-insensitive) wins. Order matters — more specific terms
// must come before more general ones, and anatomical-site keywords win
// over cell-type keywords (e.g. "ocular melanoma" → ocular_otic, not skin).
const RULES: Array<{ keyword: string; category: VetIcdOCategoryId }> = [
  // Hematopoietic (check before 'cell' / 'sarcoma' rules)
  { keyword: 'lymphoma', category: 'hematopoietic' },
  { keyword: 'leukemia', category: 'hematopoietic' },
  { keyword: 'plasmacytoma', category: 'hematopoietic' },
  { keyword: 'plasma cell', category: 'hematopoietic' },
  { keyword: 'multiple myeloma', category: 'hematopoietic' },
  { keyword: 'myeloma', category: 'hematopoietic' },
  { keyword: 'histiocytic sarcoma', category: 'hematopoietic' },

  // Urinary (check 'transitional cell' before generic 'cell')
  { keyword: 'transitional cell', category: 'urinary' },
  { keyword: 'urothelial', category: 'urinary' },
  { keyword: 'bladder', category: 'urinary' },
  { keyword: 'renal', category: 'urinary' },

  // Anatomical-site overrides — these must come BEFORE skin / soft-tissue /
  // alimentary cell-type rules so e.g. "ocular melanoma" → ocular_otic and
  // "oral squamous cell carcinoma" → alimentary.
  { keyword: 'ocular', category: 'ocular_otic' },
  { keyword: 'uveal', category: 'ocular_otic' },
  { keyword: 'ceruminous', category: 'ocular_otic' },
  { keyword: 'oral', category: 'alimentary' },

  // Bone & hard tissues
  { keyword: 'osteosarcoma', category: 'bone' },
  { keyword: 'chondrosarcoma', category: 'bone' },

  // Skin & melanocytic (check before generic 'sarcoma' / 'carcinoma')
  { keyword: 'mast cell', category: 'skin' },
  { keyword: 'melanoma', category: 'skin' },
  { keyword: 'squamous cell', category: 'skin' },
  { keyword: 'basal cell', category: 'skin' },
  { keyword: 'sebaceous', category: 'skin' },
  { keyword: 'cutaneous', category: 'skin' },

  // Soft tissue
  { keyword: 'hemangiosarcoma', category: 'soft_tissue' },
  { keyword: 'fibrosarcoma', category: 'soft_tissue' },
  { keyword: 'liposarcoma', category: 'soft_tissue' },
  { keyword: 'leiomyosarcoma', category: 'soft_tissue' },
  { keyword: 'rhabdomyosarcoma', category: 'soft_tissue' },
  { keyword: 'synovial', category: 'soft_tissue' },
  { keyword: 'soft tissue sarcoma', category: 'soft_tissue' },

  // Mammary
  { keyword: 'mammary', category: 'mammary' },

  // Genital
  { keyword: 'seminoma', category: 'genital' },
  { keyword: 'sertoli', category: 'genital' },
  { keyword: 'interstitial cell tumor', category: 'genital' },
  { keyword: 'ovarian', category: 'genital' },
  { keyword: 'uterine', category: 'genital' },
  { keyword: 'vaginal', category: 'genital' },
  { keyword: 'prostatic', category: 'genital' },
  { keyword: 'prostate', category: 'genital' },
  { keyword: 'transmissible venereal', category: 'genital' },

  // Nervous system
  { keyword: 'meningioma', category: 'nervous' },
  { keyword: 'glioma', category: 'nervous' },
  { keyword: 'astrocytoma', category: 'nervous' },
  { keyword: 'oligodendroglioma', category: 'nervous' },
  { keyword: 'schwannoma', category: 'nervous' },
  { keyword: 'ependymoma', category: 'nervous' },
  { keyword: 'choroid plexus', category: 'nervous' },

  // Respiratory
  { keyword: 'pulmonary', category: 'respiratory' },
  { keyword: 'lung', category: 'respiratory' },
  { keyword: 'nasal', category: 'respiratory' },
  { keyword: 'bronchial', category: 'respiratory' },
  { keyword: 'tracheal', category: 'respiratory' },

  // Alimentary (note: 'oral' moved up to anatomical-site overrides above)
  { keyword: 'hepatocellular', category: 'alimentary' },
  { keyword: 'hepatic', category: 'alimentary' },
  { keyword: 'pancreatic', category: 'alimentary' },
  { keyword: 'gastric', category: 'alimentary' },
  { keyword: 'intestinal', category: 'alimentary' },
  { keyword: 'colonic', category: 'alimentary' },
  { keyword: 'salivary', category: 'alimentary' },

  // Endocrine
  { keyword: 'thyroid', category: 'endocrine' },
  { keyword: 'adrenal', category: 'endocrine' },
  { keyword: 'pituitary', category: 'endocrine' },
  { keyword: 'insulinoma', category: 'endocrine' },
  { keyword: 'pheochromocytoma', category: 'endocrine' },
  { keyword: 'parathyroid', category: 'endocrine' },
];

export function classifyCancerType(cancerType: string): VetIcdOCategoryId {
  const needle = cancerType.toLowerCase();
  for (const rule of RULES) {
    if (needle.includes(rule.keyword)) return rule.category;
  }
  return 'other';
}
