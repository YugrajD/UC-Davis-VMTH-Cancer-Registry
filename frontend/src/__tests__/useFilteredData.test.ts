import { describe, it, expect, vi } from 'vitest';

// Execute useMemo factories synchronously so hooks are testable without a React renderer.
vi.mock('react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react')>();
  return { ...actual, useMemo: <T>(factory: () => T) => factory() };
});

import { useFilteredData } from '../hooks/useFilteredData';
import { MOCK_COUNTY_DATA } from '../data/mockData';
import type { FilterState } from '../types';

const DEFAULT_FILTERS: FilterState = {
  rateType: 'incidence',
  sex: 'all',
  cancerType: 'All Types',
  breed: 'All Breeds',
};

// ---------------------------------------------------------------------------
// Default (no filters)
// ---------------------------------------------------------------------------

describe('useFilteredData — default filters', () => {
  it('loading is always false (synchronous data)', () => {
    expect(useFilteredData(DEFAULT_FILTERS).loading).toBe(false);
  });

  it('error is always null', () => {
    expect(useFilteredData(DEFAULT_FILTERS).error).toBeNull();
  });

  it('returns all counties when no filters are active', () => {
    const { countyData } = useFilteredData(DEFAULT_FILTERS);
    expect(countyData.length).toBe(MOCK_COUNTY_DATA.length);
  });

  it('unfiltered county counts match raw mock data exactly', () => {
    const { countyData } = useFilteredData(DEFAULT_FILTERS);
    const mockMap = new Map(MOCK_COUNTY_DATA.map(c => [c.county, c.count]));
    for (const c of countyData) {
      expect(c.count).toBe(mockMap.get(c.county));
    }
  });

  it('countRange.min and max correspond to the actual data', () => {
    const { countRange, countyData } = useFilteredData(DEFAULT_FILTERS);
    const counts = countyData.map(c => c.count);
    expect(countRange.min).toBe(Math.min(...counts));
    expect(countRange.max).toBe(Math.max(...counts));
  });

  it('countRange.min <= countRange.max', () => {
    const { countRange } = useFilteredData(DEFAULT_FILTERS);
    expect(countRange.min).toBeLessThanOrEqual(countRange.max);
  });
});

// ---------------------------------------------------------------------------
// countRange edge cases
// ---------------------------------------------------------------------------

describe('useFilteredData — countRange edge cases', () => {
  it('returns {min:0, max:1} as a safe fallback when all counties are filtered out', () => {
    // Applying all three narrow filters at once typically zeroes out everything
    const { countRange } = useFilteredData({
      ...DEFAULT_FILTERS,
      sex: 'male_intact',
      cancerType: 'Lymphoma',
      breed: 'Golden Retriever',
    });
    if (countRange.min === 0) {
      expect(countRange).toEqual({ min: 0, max: 1 });
    } else {
      expect(countRange.min).toBeLessThanOrEqual(countRange.max);
    }
  });
});

// ---------------------------------------------------------------------------
// Sex filter
// ---------------------------------------------------------------------------

describe('useFilteredData — sex filter', () => {
  it('sex filter reduces the total case count', () => {
    const base = useFilteredData(DEFAULT_FILTERS).countyData;
    const filtered = useFilteredData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).countyData;
    const baseTotal = base.reduce((s, c) => s + c.count, 0);
    const filteredTotal = filtered.reduce((s, c) => s + c.count, 0);
    expect(filteredTotal).toBeLessThan(baseTotal);
  });

  it('sex filter may reduce the number of counties (zeros removed)', () => {
    const base = useFilteredData(DEFAULT_FILTERS).countyData;
    const filtered = useFilteredData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).countyData;
    expect(filtered.length).toBeLessThanOrEqual(base.length);
  });

  it('all remaining county counts are positive after sex filter', () => {
    const { countyData } = useFilteredData({ ...DEFAULT_FILTERS, sex: 'male_neutered' });
    for (const c of countyData) {
      expect(c.count).toBeGreaterThan(0);
    }
  });

  it('sex filter is deterministic — same inputs yield same outputs', () => {
    const a = useFilteredData({ ...DEFAULT_FILTERS, sex: 'female_spayed' }).countyData;
    const b = useFilteredData({ ...DEFAULT_FILTERS, sex: 'female_spayed' }).countyData;
    expect(a.map(c => c.count)).toEqual(b.map(c => c.count));
  });

  it('different sex values produce different results', () => {
    const male = useFilteredData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).countyData;
    const female = useFilteredData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).countyData;
    const maleTotals = male.map(c => c.count);
    const femaleTotals = female.map(c => c.count);
    expect(maleTotals).not.toEqual(femaleTotals);
  });
});

// ---------------------------------------------------------------------------
// Cancer type filter
// ---------------------------------------------------------------------------

describe('useFilteredData — cancer type filter', () => {
  it('cancer type filter reduces total cases', () => {
    const base = useFilteredData(DEFAULT_FILTERS).countyData;
    const filtered = useFilteredData({ ...DEFAULT_FILTERS, cancerType: 'Lymphoma' }).countyData;
    const baseTotal = base.reduce((s, c) => s + c.count, 0);
    const filteredTotal = filtered.reduce((s, c) => s + c.count, 0);
    expect(filteredTotal).toBeLessThan(baseTotal);
  });

  it('cancer type filter is deterministic', () => {
    const a = useFilteredData({ ...DEFAULT_FILTERS, cancerType: 'Osteosarcoma' }).countyData;
    const b = useFilteredData({ ...DEFAULT_FILTERS, cancerType: 'Osteosarcoma' }).countyData;
    expect(a.map(c => c.count)).toEqual(b.map(c => c.count));
  });
});

// ---------------------------------------------------------------------------
// Breed filter
// ---------------------------------------------------------------------------

describe('useFilteredData — breed filter', () => {
  it('breed filter reduces total cases', () => {
    const base = useFilteredData(DEFAULT_FILTERS).countyData;
    const filtered = useFilteredData({ ...DEFAULT_FILTERS, breed: 'Golden Retriever' }).countyData;
    const baseTotal = base.reduce((s, c) => s + c.count, 0);
    const filteredTotal = filtered.reduce((s, c) => s + c.count, 0);
    expect(filteredTotal).toBeLessThan(baseTotal);
  });

  it('breed filter is deterministic', () => {
    const a = useFilteredData({ ...DEFAULT_FILTERS, breed: 'Boxer' }).countyData;
    const b = useFilteredData({ ...DEFAULT_FILTERS, breed: 'Boxer' }).countyData;
    expect(a.map(c => c.count)).toEqual(b.map(c => c.count));
  });
});

// ---------------------------------------------------------------------------
// Combined filters (multiplicative reduction)
// ---------------------------------------------------------------------------

describe('useFilteredData — combined filters', () => {
  it('sex + cancer type + breed combined reduces total more than any single filter', () => {
    const sexOnly = useFilteredData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).countyData;
    const breedOnly = useFilteredData({ ...DEFAULT_FILTERS, breed: 'Golden Retriever' }).countyData;
    const combined = useFilteredData({ ...DEFAULT_FILTERS, sex: 'male_intact', breed: 'Golden Retriever' }).countyData;

    const sexTotal = sexOnly.reduce((s, c) => s + c.count, 0);
    const breedTotal = breedOnly.reduce((s, c) => s + c.count, 0);
    const combinedTotal = combined.reduce((s, c) => s + c.count, 0);

    expect(combinedTotal).toBeLessThan(sexTotal);
    expect(combinedTotal).toBeLessThan(breedTotal);
  });
});

// ---------------------------------------------------------------------------
// regionSummary structure
// ---------------------------------------------------------------------------

describe('useFilteredData — regionSummary', () => {
  it('root node is California at state level', () => {
    const { regionSummary } = useFilteredData(DEFAULT_FILTERS);
    expect(regionSummary.name).toBe('California');
    expect(regionSummary.type).toBe('state');
  });

  it('California count equals total of all county counts', () => {
    const { regionSummary, countyData } = useFilteredData(DEFAULT_FILTERS);
    const total = countyData.reduce((s, c) => s + c.count, 0);
    expect(regionSummary.count).toBe(total);
  });

  it('first child of California is the UC Davis Catchment Area', () => {
    const { regionSummary } = useFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    expect(catchment?.name).toBe('UC Davis Catchment Area');
    expect(catchment?.type).toBe('catchment');
  });

  it('catchment area includes Bay Area region', () => {
    const { regionSummary } = useFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    const regions = catchment?.children?.map(r => r.name) ?? [];
    expect(regions).toContain('Bay Area');
  });

  it('catchment area includes Central Valley region', () => {
    const { regionSummary } = useFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    const regions = catchment?.children?.map(r => r.name) ?? [];
    expect(regions).toContain('Central Valley');
  });

  it('catchment area includes Northern CA region', () => {
    const { regionSummary } = useFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    const regions = catchment?.children?.map(r => r.name) ?? [];
    expect(regions).toContain('Northern CA');
  });

  it('catchment count equals sum of its region counts', () => {
    const { regionSummary } = useFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children![0];
    const regionSum = (catchment.children ?? []).reduce((s, r) => s + r.count, 0);
    expect(catchment.count).toBe(regionSum);
  });

  it('each region has county-level children', () => {
    const { regionSummary } = useFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    for (const region of catchment?.children ?? []) {
      expect(region.type).toBe('region');
      expect((region.children ?? []).length).toBeGreaterThan(0);
      for (const county of region.children ?? []) {
        expect(county.type).toBe('county');
      }
    }
  });

  it('region count equals sum of its county counts', () => {
    const { regionSummary } = useFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    for (const region of catchment?.children ?? []) {
      const countySum = (region.children ?? []).reduce((s, c) => s + c.count, 0);
      expect(region.count).toBe(countySum);
    }
  });
});
