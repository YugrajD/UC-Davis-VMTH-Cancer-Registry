import { describe, it, expect, vi } from 'vitest';

// Mock useMemo to execute the factory synchronously so hooks are testable without
// a React rendering environment.
vi.mock('react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react')>();
  return { ...actual, useMemo: <T>(factory: () => T) => factory() };
});

import { useCancerTypesData } from '../hooks/useCancerTypesData';
import { MOCK_CANCER_TYPE_INCIDENTS } from '../data/mockData';
import type { FilterState } from '../types';

const DEFAULT_FILTERS: FilterState = {
  rateType: 'incidence',
  sex: 'all',
  cancerType: 'All Types',
  breed: 'All Breeds',
};

describe('useCancerTypesData — default filters', () => {
  it('returns all 8 cancer types when no filters are applied', () => {
    const { data } = useCancerTypesData(DEFAULT_FILTERS);
    expect(data.length).toBe(MOCK_CANCER_TYPE_INCIDENTS.length);
  });

  it('loading is always false (synchronous mock data)', () => {
    expect(useCancerTypesData(DEFAULT_FILTERS).loading).toBe(false);
  });

  it('error is always null', () => {
    expect(useCancerTypesData(DEFAULT_FILTERS).error).toBeNull();
  });

  it('each record has a cancer_type string and positive count', () => {
    const { data } = useCancerTypesData(DEFAULT_FILTERS);
    for (const r of data) {
      expect(typeof r.cancer_type).toBe('string');
      expect(r.count).toBeGreaterThan(0);
    }
  });

  it('unfiltered counts match the raw mock data exactly', () => {
    const { data } = useCancerTypesData(DEFAULT_FILTERS);
    const mockMap = new Map(MOCK_CANCER_TYPE_INCIDENTS.map(r => [r.cancer_type, r.count]));
    for (const r of data) {
      expect(r.count).toBe(mockMap.get(r.cancer_type));
    }
  });
});

describe('useCancerTypesData — cancer type filter', () => {
  it('returns only the matching type when a specific cancer type is selected', () => {
    const { data } = useCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'Lymphoma' });
    expect(data.length).toBe(1);
    expect(data[0].cancer_type).toBe('Lymphoma');
  });

  it('cancer type matching is case-insensitive', () => {
    const { data } = useCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'lymphoma' });
    expect(data.length).toBe(1);
    expect(data[0].cancer_type).toBe('Lymphoma');
  });

  it('returns all types when cancer type is not found in data', () => {
    const { data } = useCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'Nonexistent Tumor' });
    expect(data.length).toBe(MOCK_CANCER_TYPE_INCIDENTS.length);
  });

  it('partial cancer type string matches', () => {
    // "Mast" should match "Mast Cell Tumor"
    const { data } = useCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'Mast' });
    expect(data.length).toBe(1);
    expect(data[0].cancer_type).toContain('Mast');
  });
});

describe('useCancerTypesData — sex filter', () => {
  it('sex filter reduces all counts compared to unfiltered', () => {
    const unfiltered = useCancerTypesData(DEFAULT_FILTERS).data;
    const filtered = useCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).data;
    const unfilteredTotal = unfiltered.reduce((s, r) => s + r.count, 0);
    const filteredTotal = filtered.reduce((s, r) => s + r.count, 0);
    expect(filteredTotal).toBeLessThan(unfilteredTotal);
  });

  it('sex filter keeps the same number of cancer type records', () => {
    const unfiltered = useCancerTypesData(DEFAULT_FILTERS).data;
    const filtered = useCancerTypesData({ ...DEFAULT_FILTERS, sex: 'female_spayed' }).data;
    expect(filtered.length).toBe(unfiltered.length);
  });

  it('every count after sex filter is at least 1', () => {
    const { data } = useCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_neutered' });
    for (const r of data) {
      expect(r.count).toBeGreaterThanOrEqual(1);
    }
  });

  it('sex filter is deterministic — same inputs yield same outputs', () => {
    const a = useCancerTypesData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).data;
    const b = useCancerTypesData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).data;
    expect(a.map(r => r.count)).toEqual(b.map(r => r.count));
  });

  it('different sex values produce different outputs', () => {
    const male = useCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).data;
    const female = useCancerTypesData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).data;
    const maleCounts = male.map(r => r.count);
    const femaleCounts = female.map(r => r.count);
    expect(maleCounts).not.toEqual(femaleCounts);
  });
});

describe('useCancerTypesData — breed filter', () => {
  it('breed filter reduces all counts compared to unfiltered', () => {
    const unfiltered = useCancerTypesData(DEFAULT_FILTERS).data;
    const filtered = useCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Golden Retriever' }).data;
    const unfilteredTotal = unfiltered.reduce((s, r) => s + r.count, 0);
    const filteredTotal = filtered.reduce((s, r) => s + r.count, 0);
    expect(filteredTotal).toBeLessThan(unfilteredTotal);
  });

  it('every count after breed filter is at least 1', () => {
    const { data } = useCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Boxer' });
    for (const r of data) {
      expect(r.count).toBeGreaterThanOrEqual(1);
    }
  });

  it('breed filter is deterministic', () => {
    const a = useCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Poodle' }).data;
    const b = useCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Poodle' }).data;
    expect(a.map(r => r.count)).toEqual(b.map(r => r.count));
  });
});

describe('useCancerTypesData — combined filters', () => {
  it('sex + breed combined reduces counts more than either alone', () => {
    const sexOnly = useCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).data;
    const breedOnly = useCancerTypesData({ ...DEFAULT_FILTERS, breed: 'Golden Retriever' }).data;
    const combined = useCancerTypesData({ ...DEFAULT_FILTERS, sex: 'male_intact', breed: 'Golden Retriever' }).data;

    const sexTotal = sexOnly.reduce((s, r) => s + r.count, 0);
    const breedTotal = breedOnly.reduce((s, r) => s + r.count, 0);
    const combinedTotal = combined.reduce((s, r) => s + r.count, 0);

    expect(combinedTotal).toBeLessThan(sexTotal);
    expect(combinedTotal).toBeLessThan(breedTotal);
  });

  it('cancer type + sex returns one record with reduced count', () => {
    const { data } = useCancerTypesData({ ...DEFAULT_FILTERS, cancerType: 'Lymphoma', sex: 'female_spayed' });
    expect(data.length).toBe(1);
    expect(data[0].cancer_type).toBe('Lymphoma');
    const unfilteredLymphoma = MOCK_CANCER_TYPE_INCIDENTS.find(r => r.cancer_type === 'Lymphoma')!;
    expect(data[0].count).toBeLessThan(unfilteredLymphoma.count);
  });
});
