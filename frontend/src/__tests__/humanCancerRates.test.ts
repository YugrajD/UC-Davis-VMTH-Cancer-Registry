import { describe, it, expect } from 'vitest';
import {
  getHumanCancerRateMap,
  HUMAN_CANCER_SITES,
  HUMAN_CANCER_SEX_OPTIONS,
  HUMAN_CANCER_RATES,
  type HumanCancerSite,
  type HumanCancerSex,
} from '../data/humanCancerRates';

// ---------------------------------------------------------------------------
// HUMAN_CANCER_SITES constant
// ---------------------------------------------------------------------------

describe('HUMAN_CANCER_SITES', () => {
  it('contains 20 entries', () => {
    expect(HUMAN_CANCER_SITES.length).toBe(20);
  });

  it('includes All Cancer Sites', () => {
    expect(HUMAN_CANCER_SITES.some(s => s.value === 'All Cancer Sites')).toBe(true);
  });

  it('every entry has a non-empty value and label', () => {
    for (const s of HUMAN_CANCER_SITES) {
      expect(s.value.length).toBeGreaterThan(0);
      expect(s.label.length).toBeGreaterThan(0);
    }
  });

  it('values are unique', () => {
    const values = HUMAN_CANCER_SITES.map(s => s.value);
    expect(new Set(values).size).toBe(values.length);
  });
});

// ---------------------------------------------------------------------------
// HUMAN_CANCER_SEX_OPTIONS constant
// ---------------------------------------------------------------------------

describe('HUMAN_CANCER_SEX_OPTIONS', () => {
  it('contains exactly 3 options', () => {
    expect(HUMAN_CANCER_SEX_OPTIONS.length).toBe(3);
  });

  it('includes Both Sexes, Male, Female', () => {
    const values = HUMAN_CANCER_SEX_OPTIONS.map(o => o.value);
    expect(values).toContain('Both Sexes');
    expect(values).toContain('Male');
    expect(values).toContain('Female');
  });
});

// ---------------------------------------------------------------------------
// HUMAN_CANCER_RATES dataset integrity
// ---------------------------------------------------------------------------

describe('HUMAN_CANCER_RATES dataset', () => {
  it('is non-empty', () => {
    expect(HUMAN_CANCER_RATES.length).toBeGreaterThan(0);
  });

  it('every entry has a non-empty county and site', () => {
    for (const r of HUMAN_CANCER_RATES) {
      expect(r.county.length).toBeGreaterThan(0);
      expect(r.site.length).toBeGreaterThan(0);
    }
  });

  it('rate values are either null or positive numbers', () => {
    for (const r of HUMAN_CANCER_RATES) {
      if (r.rate !== null) {
        expect(r.rate).toBeGreaterThan(0);
      }
    }
  });

  it('cases values are either null or positive numbers', () => {
    for (const r of HUMAN_CANCER_RATES) {
      if (r.cases !== null) {
        expect(r.cases).toBeGreaterThan(0);
      }
    }
  });

  it('contains data for Alameda county', () => {
    expect(HUMAN_CANCER_RATES.some(r => r.county === 'Alameda')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// getHumanCancerRateMap — structure
// ---------------------------------------------------------------------------

describe('getHumanCancerRateMap — structure', () => {
  it('returns a Map', () => {
    const m = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
    expect(m).toBeInstanceOf(Map);
  });

  it('returns a non-empty map for All Cancer Sites / Both Sexes', () => {
    const m = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
    expect(m.size).toBeGreaterThan(0);
  });

  it('keys are lowercase county names', () => {
    const m = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
    for (const key of m.keys()) {
      expect(key).toBe(key.toLowerCase());
    }
  });

  it('each value has rate and cases properties', () => {
    const m = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
    for (const val of m.values()) {
      expect('rate' in val).toBe(true);
      expect('cases' in val).toBe(true);
    }
  });

  it('returns empty map for unknown site', () => {
    const m = getHumanCancerRateMap('Unknown Site' as HumanCancerSite, 'Both Sexes');
    expect(m.size).toBe(0);
  });

  it('returns empty map for unknown sex', () => {
    const m = getHumanCancerRateMap('All Cancer Sites', 'Unknown' as HumanCancerSex);
    expect(m.size).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// getHumanCancerRateMap — correctness
// ---------------------------------------------------------------------------

describe('getHumanCancerRateMap — correctness', () => {
  it('contains alameda for All Cancer Sites / Both Sexes', () => {
    const m = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
    expect(m.has('alameda')).toBe(true);
  });

  it('alameda rate for All Cancer Sites / Both Sexes matches source data', () => {
    const m = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
    const entry = m.get('alameda');
    const source = HUMAN_CANCER_RATES.find(
      r => r.county === 'Alameda' && r.site === 'All Cancer Sites' && r.sex === 'Both Sexes',
    );
    expect(entry?.rate).toBe(source?.rate);
  });

  it('Male and Female maps differ from Both Sexes', () => {
    const both = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
    const male = getHumanCancerRateMap('All Cancer Sites', 'Male');
    const alamedaBoth = both.get('alameda')?.rate;
    const alamedaMale = male.get('alameda')?.rate;
    expect(alamedaBoth).not.toBe(alamedaMale);
  });

  it('different sites return different data', () => {
    const all = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
    const lung = getHumanCancerRateMap('Lung & Bronchus', 'Both Sexes');
    const allRate = all.get('alameda')?.rate;
    const lungRate = lung.get('alameda')?.rate;
    expect(allRate).not.toBe(lungRate);
  });

  it('is deterministic — same inputs produce same map', () => {
    const a = getHumanCancerRateMap('Lung & Bronchus', 'Female');
    const b = getHumanCancerRateMap('Lung & Bronchus', 'Female');
    expect(a.size).toBe(b.size);
    for (const [k, v] of a) {
      expect(b.get(k)).toEqual(v);
    }
  });

  it('Prostate cancer has no Female data', () => {
    const m = getHumanCancerRateMap('Prostate', 'Female');
    expect(m.size).toBe(0);
  });

  it('Breast (Female) cancer returns data for Female', () => {
    const m = getHumanCancerRateMap('Breast (Female)', 'Female');
    expect(m.size).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// getHumanCancerRateMap — default parameters
// ---------------------------------------------------------------------------

describe('getHumanCancerRateMap — default parameters', () => {
  it('calling with no args uses All Cancer Sites / Both Sexes defaults', () => {
    const withDefaults = getHumanCancerRateMap();
    const explicit = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
    expect(withDefaults.size).toBe(explicit.size);
    for (const [k, v] of explicit) {
      expect(withDefaults.get(k)).toEqual(v);
    }
  });
});
