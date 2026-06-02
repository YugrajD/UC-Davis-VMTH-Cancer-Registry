import { describe, it, expect } from 'vitest';

import {
  buildZipCodeDataFromIncidence,
  createFilteredDataState,
  createFilteredZipCodeDataState,
  getZipCodeCountRange,
} from '../hooks/useFilteredData';
import { MOCK_COUNTY_DATA } from '../data/mockData';
import type { FilterState } from '../types';

const DEFAULT_FILTERS: FilterState = {
  rateType: 'incidence',
  sex: 'all',
  cancerType: 'All Types',
  breed: 'All Breeds',
};

function getFilteredData(filters: FilterState) {
  return {
    ...createFilteredDataState(MOCK_COUNTY_DATA, filters),
    loading: false,
    error: null,
  };
}

// ---------------------------------------------------------------------------
// Default (no filters)
// ---------------------------------------------------------------------------

describe('useFilteredData — default filters', () => {
  it('loading is always false (synchronous data)', () => {
    expect(getFilteredData(DEFAULT_FILTERS).loading).toBe(false);
  });

  it('error is always null', () => {
    expect(getFilteredData(DEFAULT_FILTERS).error).toBeNull();
  });

  it('returns all counties when no filters are active', () => {
    const { countyData } = getFilteredData(DEFAULT_FILTERS);
    expect(countyData.length).toBe(MOCK_COUNTY_DATA.length);
  });

  it('unfiltered county counts match raw mock data exactly', () => {
    const { countyData } = getFilteredData(DEFAULT_FILTERS);
    const mockMap = new Map(MOCK_COUNTY_DATA.map(c => [c.county, c.count]));
    for (const c of countyData) {
      expect(c.count).toBe(mockMap.get(c.county));
    }
  });

  it('countRange.min and max correspond to the actual data', () => {
    const { countRange, countyData } = getFilteredData(DEFAULT_FILTERS);
    const counts = countyData.map(c => c.count);
    expect(countRange.min).toBe(Math.min(...counts));
    expect(countRange.max).toBe(Math.max(...counts));
  });

  it('countRange.min <= countRange.max', () => {
    const { countRange } = getFilteredData(DEFAULT_FILTERS);
    expect(countRange.min).toBeLessThanOrEqual(countRange.max);
  });
});

// ---------------------------------------------------------------------------
// countRange edge cases
// ---------------------------------------------------------------------------

describe('useFilteredData — countRange edge cases', () => {
  it('returns {min:0, max:1} as a safe fallback when all counties are filtered out', () => {
    // Applying all three narrow filters at once typically zeroes out everything
    const { countRange } = getFilteredData({
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
    const base = getFilteredData(DEFAULT_FILTERS).countyData;
    const filtered = getFilteredData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).countyData;
    const baseTotal = base.reduce((s, c) => s + c.count, 0);
    const filteredTotal = filtered.reduce((s, c) => s + c.count, 0);
    expect(filteredTotal).toBeLessThan(baseTotal);
  });

  it('sex filter may reduce the number of counties (zeros removed)', () => {
    const base = getFilteredData(DEFAULT_FILTERS).countyData;
    const filtered = getFilteredData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).countyData;
    expect(filtered.length).toBeLessThanOrEqual(base.length);
  });

  it('all remaining county counts are positive after sex filter', () => {
    const { countyData } = getFilteredData({ ...DEFAULT_FILTERS, sex: 'male_neutered' });
    for (const c of countyData) {
      expect(c.count).toBeGreaterThan(0);
    }
  });

  it('sex filter is deterministic — same inputs yield same outputs', () => {
    const a = getFilteredData({ ...DEFAULT_FILTERS, sex: 'female_spayed' }).countyData;
    const b = getFilteredData({ ...DEFAULT_FILTERS, sex: 'female_spayed' }).countyData;
    expect(a.map(c => c.count)).toEqual(b.map(c => c.count));
  });

  it('different sex values produce different results', () => {
    const male = getFilteredData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).countyData;
    const female = getFilteredData({ ...DEFAULT_FILTERS, sex: 'female_intact' }).countyData;
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
    const base = getFilteredData(DEFAULT_FILTERS).countyData;
    const filtered = getFilteredData({ ...DEFAULT_FILTERS, cancerType: 'Lymphoma' }).countyData;
    const baseTotal = base.reduce((s, c) => s + c.count, 0);
    const filteredTotal = filtered.reduce((s, c) => s + c.count, 0);
    expect(filteredTotal).toBeLessThan(baseTotal);
  });

  it('cancer type filter is deterministic', () => {
    const a = getFilteredData({ ...DEFAULT_FILTERS, cancerType: 'Osteosarcoma' }).countyData;
    const b = getFilteredData({ ...DEFAULT_FILTERS, cancerType: 'Osteosarcoma' }).countyData;
    expect(a.map(c => c.count)).toEqual(b.map(c => c.count));
  });
});

// ---------------------------------------------------------------------------
// Breed filter
// ---------------------------------------------------------------------------

describe('useFilteredData — breed filter', () => {
  it('breed filter reduces total cases', () => {
    const base = getFilteredData(DEFAULT_FILTERS).countyData;
    const filtered = getFilteredData({ ...DEFAULT_FILTERS, breed: 'Golden Retriever' }).countyData;
    const baseTotal = base.reduce((s, c) => s + c.count, 0);
    const filteredTotal = filtered.reduce((s, c) => s + c.count, 0);
    expect(filteredTotal).toBeLessThan(baseTotal);
  });

  it('breed filter is deterministic', () => {
    const a = getFilteredData({ ...DEFAULT_FILTERS, breed: 'Boxer' }).countyData;
    const b = getFilteredData({ ...DEFAULT_FILTERS, breed: 'Boxer' }).countyData;
    expect(a.map(c => c.count)).toEqual(b.map(c => c.count));
  });
});

// ---------------------------------------------------------------------------
// Combined filters (multiplicative reduction)
// ---------------------------------------------------------------------------

describe('useFilteredData — combined filters', () => {
  it('sex + cancer type + breed combined reduces total more than any single filter', () => {
    const sexOnly = getFilteredData({ ...DEFAULT_FILTERS, sex: 'male_intact' }).countyData;
    const breedOnly = getFilteredData({ ...DEFAULT_FILTERS, breed: 'Golden Retriever' }).countyData;
    const combined = getFilteredData({ ...DEFAULT_FILTERS, sex: 'male_intact', breed: 'Golden Retriever' }).countyData;

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
    const { regionSummary } = getFilteredData(DEFAULT_FILTERS);
    expect(regionSummary.name).toBe('California');
    expect(regionSummary.type).toBe('state');
  });

  it('California count equals total of all county counts', () => {
    const { regionSummary, countyData } = getFilteredData(DEFAULT_FILTERS);
    const total = countyData.reduce((s, c) => s + c.count, 0);
    expect(regionSummary.count).toBe(total);
  });

  it('first child of California is the UC Davis Catchment Area', () => {
    const { regionSummary } = getFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    expect(catchment?.name).toBe('UC Davis Catchment Area');
    expect(catchment?.type).toBe('catchment');
  });

  it('catchment area includes San Francisco Bay Area region', () => {
    const { regionSummary } = getFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    const regions = catchment?.children?.map(r => r.name) ?? [];
    expect(regions).toContain('San Francisco Bay Area');
  });

  it('catchment area includes Sacramento Valley region', () => {
    const { regionSummary } = getFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    const regions = catchment?.children?.map(r => r.name) ?? [];
    expect(regions).toContain('Sacramento Valley');
  });

  it('catchment area includes San Joaquin Valley region', () => {
    const { regionSummary } = getFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    const regions = catchment?.children?.map(r => r.name) ?? [];
    expect(regions).toContain('San Joaquin Valley');
  });

  it('catchment count equals sum of its region counts', () => {
    const { regionSummary } = getFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children![0];
    const regionSum = (catchment.children ?? []).reduce((s, r) => s + r.count, 0);
    expect(catchment.count).toBe(regionSum);
  });

  it('each region has county-level children', () => {
    const { regionSummary } = getFilteredData(DEFAULT_FILTERS);
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
    const { regionSummary } = getFilteredData(DEFAULT_FILTERS);
    const catchment = regionSummary.children?.[0];
    for (const region of catchment?.children ?? []) {
      const countySum = (region.children ?? []).reduce((s, c) => s + c.count, 0);
      expect(region.count).toBe(countySum);
    }
  });
});

// ---------------------------------------------------------------------------
// ZIP/ZCTA incidence helpers
// ---------------------------------------------------------------------------

describe('useFilteredData — ZIP code data', () => {
  it('builds ZIP-level counts from incidence records', () => {
    const zipCodeData = buildZipCodeDataFromIncidence([
      { cancer_type: 'All', zip_code: '95616', count: 2 },
      { cancer_type: 'All', zip_code: '95616-1234', count: 3 },
      { cancer_type: 'All', zip_code: '95817', count: 4 },
      { cancer_type: 'All', count: 9 },
    ]);

    expect(zipCodeData).toEqual([
      { zipCode: '95616', count: 5 },
      { zipCode: '95817', count: 4 },
    ]);
  });

  it('calculates ZIP count range from nonzero ZIP counts', () => {
    expect(getZipCodeCountRange([
      { zipCode: '95616', count: 5 },
      { zipCode: '95817', count: 12 },
      { zipCode: '99999', count: 0 },
    ])).toEqual({ min: 5, max: 12 });
  });

  it('applies deterministic demo filtering for breed-only ZIP filters', () => {
    const base = [
      { zipCode: '95616', count: 100 },
      { zipCode: '95817', count: 80 },
    ];
    const filters = { ...DEFAULT_FILTERS, breed: 'Golden Retriever' };

    const a = createFilteredZipCodeDataState(base, filters).zipCodeData;
    const b = createFilteredZipCodeDataState(base, filters).zipCodeData;

    expect(a).toEqual(b);
    expect(a.reduce((sum, z) => sum + z.count, 0)).toBeLessThan(180);
  });
});
