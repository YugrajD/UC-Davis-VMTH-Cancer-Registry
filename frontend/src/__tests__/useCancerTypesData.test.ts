import { describe, it, expect } from 'vitest';

import { applyCancerTypeDemoFilters } from '../hooks/useCancerTypesData';
import { MOCK_CANCER_TYPE_INCIDENTS } from '../data/mockData';
import type { FilterState } from '../types';

const DEFAULT_FILTERS: FilterState = {
  rateType: 'incidence',
  sex: 'all',
  ageGroup: 'all',
  cancerType: 'All Types',
  breed: 'All Breeds',
};

function getCancerTypesData(filters: FilterState) {
  return {
    data: applyCancerTypeDemoFilters(MOCK_CANCER_TYPE_INCIDENTS, filters),
    loading: false,
    error: null,
  };
}

describe('useCancerTypesData — default filters', () => {
  it('returns all 8 cancer types when no filters are applied', () => {
    const { data } = getCancerTypesData(DEFAULT_FILTERS);
    expect(data.length).toBe(MOCK_CANCER_TYPE_INCIDENTS.length);
  });

  it('loading is always false (synchronous mock data)', () => {
    expect(getCancerTypesData(DEFAULT_FILTERS).loading).toBe(false);
  });

  it('error is always null', () => {
    expect(getCancerTypesData(DEFAULT_FILTERS).error).toBeNull();
  });

  it('each record has a cancer_type string and positive count', () => {
    const { data } = getCancerTypesData(DEFAULT_FILTERS);
    for (const r of data) {
      expect(typeof r.cancer_type).toBe('string');
      expect(r.count).toBeGreaterThan(0);
    }
  });

  it('unfiltered counts match the raw mock data exactly', () => {
    const { data } = getCancerTypesData(DEFAULT_FILTERS);
    const mockMap = new Map(MOCK_CANCER_TYPE_INCIDENTS.map(r => [r.cancer_type, r.count]));
    for (const r of data) {
      expect(r.count).toBe(mockMap.get(r.cancer_type));
    }
  });
});

describe('useCancerTypesData — cancer type filter', () => {
  it('returns only the matching type when a specific cancer type is selected', () => {
    const { data } = getCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'Lymphoma' });
    expect(data.length).toBe(1);
    expect(data[0].cancer_type).toBe('Lymphoma');
  });

  it('cancer type matching is case-insensitive', () => {
    const { data } = getCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'lymphoma' });
    expect(data.length).toBe(1);
    expect(data[0].cancer_type).toBe('Lymphoma');
  });

  it('returns all types when cancer type is not found in data', () => {
    const { data } = getCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'Nonexistent Tumor' });
    expect(data.length).toBe(MOCK_CANCER_TYPE_INCIDENTS.length);
  });

  it('partial cancer type string matches', () => {
    // "Mast" should match "Mast Cell Tumor"
    const { data } = getCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'Mast' });
    expect(data.length).toBe(1);
    expect(data[0].cancer_type).toContain('Mast');
  });
});

describe('useCancerTypesData — sex filter', () => {
  it('sex filter reduces all counts compared to unfiltered', () => {
    const unfiltered = getCancerTypesData(DEFAULT_FILTERS).data;
    const filtered = getCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).data;
    const unfilteredTotal = unfiltered.reduce((s, r) => s + r.count, 0);
    const filteredTotal = filtered.reduce((s, r) => s + r.count, 0);
    expect(filteredTotal).toBeLessThan(unfilteredTotal);
  });

  it('sex filter keeps the same number of cancer type records', () => {
    const unfiltered = getCancerTypesData(DEFAULT_FILTERS).data;
    const filtered = getCancerTypesData({ ...DEFAULT_FILTERS, sex: 'female_spayed' }).data;
    expect(filtered.length).toBe(unfiltered.length);
  });

  it('every count after sex filter is at least 1', () => {
    const { data } = getCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_neutered' });
    for (const r of data) {
      expect(r.count).toBeGreaterThanOrEqual(1);
    }
  });

  it('sex filter is deterministic — same inputs yield same outputs', () => {
    const a = getCancerTypesData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).data;
    const b = getCancerTypesData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).data;
    expect(a.map(r => r.count)).toEqual(b.map(r => r.count));
  });

  it('different sex values produce different outputs', () => {
    const male = getCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).data;
    const female = getCancerTypesData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).data;
    const maleCounts = male.map(r => r.count);
    const femaleCounts = female.map(r => r.count);
    expect(maleCounts).not.toEqual(femaleCounts);
  });
});

describe('useCancerTypesData — breed filter', () => {
  it('breed filter reduces all counts compared to unfiltered', () => {
    const unfiltered = getCancerTypesData(DEFAULT_FILTERS).data;
    const filtered = getCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Golden Retriever' }).data;
    const unfilteredTotal = unfiltered.reduce((s, r) => s + r.count, 0);
    const filteredTotal = filtered.reduce((s, r) => s + r.count, 0);
    expect(filteredTotal).toBeLessThan(unfilteredTotal);
  });

  it('every count after breed filter is at least 1', () => {
    const { data } = getCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Boxer' });
    for (const r of data) {
      expect(r.count).toBeGreaterThanOrEqual(1);
    }
  });

  it('breed filter is deterministic', () => {
    const a = getCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Poodle' }).data;
    const b = getCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Poodle' }).data;
    expect(a.map(r => r.count)).toEqual(b.map(r => r.count));
  });
});

describe('useCancerTypesData — combined filters', () => {
  it('sex + breed combined reduces counts more than either alone', () => {
    const sexOnly = getCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).data;
    const breedOnly = getCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Golden Retriever' }).data;
    const combined = getCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_intact', breed: 'Golden Retriever' }).data;

    const sexTotal = sexOnly.reduce((s, r) => s + r.count, 0);
    const breedTotal = breedOnly.reduce((s, r) => s + r.count, 0);
    const combinedTotal = combined.reduce((s, r) => s + r.count, 0);

    expect(combinedTotal).toBeLessThan(sexTotal);
    expect(combinedTotal).toBeLessThan(breedTotal);
  });

  it('cancer type + sex returns one record with reduced count', () => {
    const { data } = getCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'Lymphoma', sex: 'female_spayed' });
    expect(data.length).toBe(1);
    expect(data[0].cancer_type).toBe('Lymphoma');
    const unfilteredLymphoma = MOCK_CANCER_TYPE_INCIDENTS.find(r => r.cancer_type === 'Lymphoma')!;
    expect(data[0].count).toBeLessThan(unfilteredLymphoma.count);
  });
});
