import { describe, it, expect } from 'vitest';
import {
  CANCER_TYPES,
  BREEDS,
  SEX_OPTIONS,
  RATE_OPTIONS,
  CES_INDICATORS,
  TABS,
} from '../types';

// ---------------------------------------------------------------------------
// CANCER_TYPES
// ---------------------------------------------------------------------------

describe('CANCER_TYPES', () => {
  it('first entry is "All Types" (the default/reset value)', () => {
    expect(CANCER_TYPES[0]).toBe('All Types');
  });

  it('contains at least 2 specific cancer types', () => {
    const specific = CANCER_TYPES.filter(t => t !== 'All Types');
    expect(specific.length).toBeGreaterThanOrEqual(2);
  });

  it('includes a lymphoma group', () => {
    expect(CANCER_TYPES).toContain('Malignant lymphomas, NOS or diffuse');
  });

  it('includes an osseous/bone tumor group', () => {
    expect(CANCER_TYPES).toContain('Osseous and chondromatous neoplasms');
  });

  it('all entries are non-empty strings', () => {
    for (const t of CANCER_TYPES) {
      expect(typeof t).toBe('string');
      expect(t.length).toBeGreaterThan(0);
    }
  });

  it('entries are unique', () => {
    expect(new Set(CANCER_TYPES).size).toBe(CANCER_TYPES.length);
  });
});

// ---------------------------------------------------------------------------
// BREEDS
// ---------------------------------------------------------------------------

describe('BREEDS', () => {
  it('first entry is "All Breeds" (the default/reset value)', () => {
    expect(BREEDS[0]).toBe('All Breeds');
  });

  it('contains at least 5 specific breeds', () => {
    const specific = BREEDS.filter(b => b !== 'All Breeds');
    expect(specific.length).toBeGreaterThanOrEqual(5);
  });

  it('includes Golden Retriever', () => {
    expect(BREEDS).toContain('Golden Retriever');
  });

  it('includes Labrador Retriever', () => {
    expect(BREEDS).toContain('Labrador Retriever');
  });

  it('all entries are non-empty strings', () => {
    for (const b of BREEDS) {
      expect(typeof b).toBe('string');
      expect(b.length).toBeGreaterThan(0);
    }
  });

  it('entries are unique', () => {
    expect(new Set(BREEDS).size).toBe(BREEDS.length);
  });
});

// ---------------------------------------------------------------------------
// SEX_OPTIONS
// ---------------------------------------------------------------------------

describe('SEX_OPTIONS', () => {
  it('first option value is "all" (the default)', () => {
    expect(SEX_OPTIONS[0].value).toBe('all');
  });

  it('contains exactly 5 options (all + 4 neuter statuses)', () => {
    expect(SEX_OPTIONS.length).toBe(5);
  });

  it('covers all four neuter status combinations', () => {
    const values = SEX_OPTIONS.map(o => o.value);
    expect(values).toContain('male_intact');
    expect(values).toContain('male_neutered');
    expect(values).toContain('female_intact');
    expect(values).toContain('female_spayed');
  });

  it('every option has a non-empty label', () => {
    for (const o of SEX_OPTIONS) {
      expect(typeof o.label).toBe('string');
      expect(o.label.length).toBeGreaterThan(0);
    }
  });

  it('option values are unique', () => {
    const values = SEX_OPTIONS.map(o => o.value);
    expect(new Set(values).size).toBe(values.length);
  });
});

// ---------------------------------------------------------------------------
// RATE_OPTIONS
// ---------------------------------------------------------------------------

describe('RATE_OPTIONS', () => {
  it('contains incidence and mortality options', () => {
    const values = RATE_OPTIONS.map(o => o.value);
    expect(values).toContain('incidence');
    expect(values).toContain('mortality');
  });

  it('every option has a non-empty label', () => {
    for (const o of RATE_OPTIONS) {
      expect(typeof o.label).toBe('string');
      expect(o.label.length).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// CES_INDICATORS
// ---------------------------------------------------------------------------

describe('CES_INDICATORS', () => {
  it('contains 24 CalEnviroScreen indicators', () => {
    expect(CES_INDICATORS.length).toBe(24);
  });

  it('first indicator is ces_score (the default)', () => {
    expect(CES_INDICATORS[0].value).toBe('ces_score');
  });

  it('includes the main environmental indicators', () => {
    const values = CES_INDICATORS.map(i => i.value);
    expect(values).toContain('ozone');
    expect(values).toContain('pm25');
    expect(values).toContain('pesticides');
    expect(values).toContain('traffic');
  });

  it('includes health/socioeconomic indicators', () => {
    const values = CES_INDICATORS.map(i => i.value);
    expect(values).toContain('asthma');
    expect(values).toContain('poverty');
    expect(values).toContain('unemployment');
  });

  it('every indicator has a non-empty human-readable label', () => {
    for (const i of CES_INDICATORS) {
      expect(typeof i.label).toBe('string');
      expect(i.label.length).toBeGreaterThan(0);
    }
  });

  it('labels contain no underscores (they are human-readable)', () => {
    for (const i of CES_INDICATORS) {
      expect(i.label).not.toContain('_');
    }
  });

  it('indicator values are unique', () => {
    const values = CES_INDICATORS.map(i => i.value);
    expect(new Set(values).size).toBe(values.length);
  });
});

// ---------------------------------------------------------------------------
// TABS
// ---------------------------------------------------------------------------

describe('TABS', () => {
  it('contains the main dashboard tabs', () => {
    const ids = TABS.map(t => t.id);
    expect(ids).toContain('overview');
    expect(ids).toContain('breed-disparities');
    expect(ids).toContain('cancer-types');
    expect(ids).toContain('cancer-by-age');
    expect(ids).toContain('analysis');
  });

  it('contains the data upload tab', () => {
    const ids = TABS.map(t => t.id);
    expect(ids).toContain('data-upload');
  });

  it('contains the review and admin tabs', () => {
    const ids = TABS.map(t => t.id);
    expect(ids).toContain('review-queue');
    expect(ids).toContain('diagnosis-review');
    expect(ids).toContain('user-management');
  });

  it('contains exactly 9 tabs', () => {
    expect(TABS.length).toBe(9);
  });

  it('every tab has a non-empty label', () => {
    for (const t of TABS) {
      expect(typeof t.label).toBe('string');
      expect(t.label.length).toBeGreaterThan(0);
    }
  });

  it('tab IDs are unique', () => {
    const ids = TABS.map(t => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
