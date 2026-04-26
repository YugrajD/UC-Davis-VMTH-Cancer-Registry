import { describe, it, expect } from 'vitest';
import {
  VET_ICD_O_CATEGORIES,
  classifyCancerType,
  type VetIcdOCategoryId,
} from '../data/vetIcdOCategories';

// ---------------------------------------------------------------------------
// Category list shape
// ---------------------------------------------------------------------------

describe('VET_ICD_O_CATEGORIES', () => {
  it('exposes the 12 paper-defined categories plus an "other" bucket', () => {
    expect(VET_ICD_O_CATEGORIES).toHaveLength(13);
  });

  it('includes the "other" fallback category', () => {
    const ids = VET_ICD_O_CATEGORIES.map(c => c.id);
    expect(ids).toContain('other');
  });

  it('every entry has a non-empty label', () => {
    for (const c of VET_ICD_O_CATEGORIES) {
      expect(c.label).toBeTruthy();
      expect(typeof c.label).toBe('string');
    }
  });

  it('category ids are unique', () => {
    const ids = VET_ICD_O_CATEGORIES.map(c => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('exposes all 12 organ-system ids from the paper', () => {
    const ids = new Set(VET_ICD_O_CATEGORIES.map(c => c.id));
    const required: VetIcdOCategoryId[] = [
      'bone',
      'soft_tissue',
      'skin',
      'genital',
      'nervous',
      'respiratory',
      'hematopoietic',
      'ocular_otic',
      'alimentary',
      'urinary',
      'mammary',
      'endocrine',
    ];
    for (const id of required) {
      expect(ids.has(id)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// classifyCancerType — common entities used in MOCK data
// ---------------------------------------------------------------------------

describe('classifyCancerType — common cancer types', () => {
  // These match the bar-chart labels in the existing MOCK data.
  const cases: Array<[string, VetIcdOCategoryId]> = [
    ['Lymphoma', 'hematopoietic'],
    ['Mast Cell Tumor', 'skin'],
    ['Hemangiosarcoma', 'soft_tissue'],
    ['Osteosarcoma', 'bone'],
    ['Melanoma', 'skin'],
    ['Transitional Cell Carcinoma', 'urinary'],
    ['Squamous Cell Carcinoma', 'skin'],
    ['Fibrosarcoma', 'soft_tissue'],
  ];

  it.each(cases)('classifies %s as %s', (input, expected) => {
    expect(classifyCancerType(input)).toBe(expected);
  });
});

// ---------------------------------------------------------------------------
// Case-insensitive substring matching
// ---------------------------------------------------------------------------

describe('classifyCancerType — case-insensitive matching', () => {
  it('matches lowercase input', () => {
    expect(classifyCancerType('lymphoma')).toBe('hematopoietic');
  });

  it('matches uppercase input', () => {
    expect(classifyCancerType('OSTEOSARCOMA')).toBe('bone');
  });

  it('matches mixed-case input', () => {
    expect(classifyCancerType('MaSt CeLl TuMoR')).toBe('skin');
  });

  it('matches when the keyword is a prefix', () => {
    expect(classifyCancerType('Hemangiosarcoma, splenic')).toBe('soft_tissue');
  });

  it('matches when the keyword is in the middle', () => {
    expect(classifyCancerType('Splenic hemangiosarcoma with metastases')).toBe('soft_tissue');
  });

  it('matches when the keyword is a suffix', () => {
    expect(classifyCancerType('Cutaneous mast cell tumor')).toBe('skin');
  });
});

// ---------------------------------------------------------------------------
// Rule precedence — specific terms must win over general ones
// ---------------------------------------------------------------------------

describe('classifyCancerType — rule precedence', () => {
  it('"Transitional Cell" wins over a generic "cell" rule', () => {
    // If we matched "cell" first, this would fall into skin (mast cell, etc.).
    expect(classifyCancerType('Transitional Cell Carcinoma')).toBe('urinary');
  });

  it('"Histiocytic Sarcoma" maps to hematopoietic, not soft tissue', () => {
    expect(classifyCancerType('Histiocytic sarcoma')).toBe('hematopoietic');
  });

  it('"Plasmacytoma" maps to hematopoietic', () => {
    expect(classifyCancerType('Cutaneous plasmacytoma')).toBe('hematopoietic');
  });
});

// ---------------------------------------------------------------------------
// Multi-organ entries (bucketed by most common primary site in dogs)
// ---------------------------------------------------------------------------

describe('classifyCancerType — entries with multiple possible sites', () => {
  it('Mast Cell Tumor → skin (most common primary site in dogs)', () => {
    expect(classifyCancerType('Mast Cell Tumor')).toBe('skin');
  });

  it('Melanoma → skin (rule precedence even though oral/ocular exist)', () => {
    expect(classifyCancerType('Melanoma')).toBe('skin');
  });

  it('Squamous Cell Carcinoma → skin', () => {
    expect(classifyCancerType('Squamous Cell Carcinoma')).toBe('skin');
  });
});

// ---------------------------------------------------------------------------
// Coverage: at least one entry per organ-system category
// ---------------------------------------------------------------------------

describe('classifyCancerType — coverage of every organ system', () => {
  const samples: Array<[string, VetIcdOCategoryId]> = [
    ['Osteosarcoma', 'bone'],
    ['Soft tissue sarcoma', 'soft_tissue'],
    ['Cutaneous mast cell tumor', 'skin'],
    ['Seminoma', 'genital'],
    ['Meningioma', 'nervous'],
    ['Pulmonary carcinoma', 'respiratory'],
    ['B-cell lymphoma', 'hematopoietic'],
    ['Ocular melanoma', 'ocular_otic'],
    ['Hepatocellular carcinoma', 'alimentary'],
    ['Bladder transitional cell carcinoma', 'urinary'],
    ['Mammary adenocarcinoma', 'mammary'],
    ['Thyroid carcinoma', 'endocrine'],
  ];

  it.each(samples)('classifies %s as %s', (input, expected) => {
    expect(classifyCancerType(input)).toBe(expected);
  });
});

// ---------------------------------------------------------------------------
// Fallback behaviour
// ---------------------------------------------------------------------------

describe('classifyCancerType — "other" fallback', () => {
  it('returns "other" for an unknown cancer name', () => {
    expect(classifyCancerType('Made-up unclassifiable tumor')).toBe('other');
  });

  it('returns "other" for an empty string', () => {
    expect(classifyCancerType('')).toBe('other');
  });

  it('returns "other" for "Unknown"', () => {
    expect(classifyCancerType('Unknown')).toBe('other');
  });

  it('returns "other" for whitespace-only input', () => {
    expect(classifyCancerType('   ')).toBe('other');
  });
});
