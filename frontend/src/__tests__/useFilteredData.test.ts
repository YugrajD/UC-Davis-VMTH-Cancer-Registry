import { describe, it, expect } from 'vitest';

import {
  buildCountyDataFromPCCP,
  buildZipCodeDataFromIncidence,
  buildZipCodeDataFromPCCP,
  createFilteredDataState,
  createFilteredZipCodeDataState,
  generateRegionSummary,
  getCountRangeForRate,
  getZipCodeCountRange,
  valueForRate,
} from '../hooks/useFilteredData';
import { MOCK_COUNTY_DATA } from '../data/mockData';
import type { CountyData, FilterState } from '../types';

const DEFAULT_FILTERS: FilterState = {
  rateType: 'pccp',
  sex: 'all',
  ageGroup: 'all',
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

  it('builds ZIP-level PCCP data (numerator/denominator/pccp) from the real endpoint response', () => {
    const zipCodeData = buildZipCodeDataFromPCCP({
      data: [
        { zip_code: '95616', cancer_patients: 40, total_patients: 100, pccp: 40 },
        { zip_code: '95817', cancer_patients: 5, total_patients: 20, pccp: 25 },
      ],
      overall_cancer_patients: 45,
      overall_total_patients: 120,
      overall_pccp: 37.5,
    });

    expect(zipCodeData).toEqual([
      { zipCode: '95616', count: 40, casePatients: 40, totalPatients: 100 },
      { zipCode: '95817', count: 25, casePatients: 5, totalPatients: 20 },
    ]);
  });
});

// ---------------------------------------------------------------------------
// buildCountyDataFromPCCP / generateRegionSummary numerator+denominator
// ---------------------------------------------------------------------------

describe('useFilteredData — PCCP numerator/denominator aggregation', () => {
  it('buildCountyDataFromPCCP carries numerator and denominator through to CountyData', () => {
    const { countyData } = buildCountyDataFromPCCP({
      data: [
        { county: 'Yolo', cancer_patients: 40, total_patients: 100, pccp: 40 },
      ],
      overall_cancer_patients: 40,
      overall_total_patients: 100,
      overall_pccp: 40,
    });

    expect(countyData[0].casePatients).toBe(40);
    expect(countyData[0].totalPatients).toBe(100);
    expect(countyData[0].count).toBe(40); // count holds the PCCP percentage
  });

  it('region-level PCCP is sum(numerator)/sum(denominator), not an average of county PCCPs', () => {
    // Two counties in the same region with very different cohort sizes and PCCPs.
    // A naive average of (80% + 20%)/2 = 50% would be wrong; the correct
    // sum-based aggregate is 90/200 = 45%.
    const countyData: CountyData[] = [
      { county: 'Small County', region: 'Sacramento Valley', count: 80, fips: '', casePatients: 8, totalPatients: 10 },
      { county: 'Big County', region: 'Sacramento Valley', count: 20, fips: '', casePatients: 82, totalPatients: 190 },
    ];

    const summary = generateRegionSummary(countyData);
    const catchment = summary.children![0];
    const region = catchment.children!.find(r => r.name === 'Sacramento Valley')!;

    expect(region.casePatients).toBe(90);
    expect(region.totalPatients).toBe(200);
    expect(region.count).toBeCloseTo(45, 5);
  });

  it('numerator and denominator sum correctly up to the state (California) level', () => {
    const countyData: CountyData[] = [
      { county: 'Yolo', region: 'Sacramento Valley', count: 40, fips: '', casePatients: 40, totalPatients: 100 },
      { county: 'Marin', region: 'San Francisco Bay Area', count: 50, fips: '', casePatients: 10, totalPatients: 20 },
    ];

    const summary = generateRegionSummary(countyData);

    expect(summary.casePatients).toBe(50);
    expect(summary.totalPatients).toBe(120);
    expect(summary.count).toBeCloseTo((50 / 120) * 100, 5);
  });
});

// ---------------------------------------------------------------------------
// valueForRate / getCountRangeForRate — Rate filter (PCCP/numerator/denominator)
// ---------------------------------------------------------------------------

describe('useFilteredData — valueForRate', () => {
  const county: CountyData = {
    county: 'Yolo', region: 'Sacramento Valley', count: 40, fips: '', casePatients: 40, totalPatients: 100,
  };

  it('pccp mode returns the count field', () => {
    expect(valueForRate(county, 'pccp')).toBe(40);
  });

  it('numerator mode returns casePatients', () => {
    expect(valueForRate(county, 'numerator')).toBe(40);
  });

  it('denominator mode returns totalPatients', () => {
    expect(valueForRate(county, 'denominator')).toBe(100);
  });

  it('returns 0 for undefined source', () => {
    expect(valueForRate(undefined, 'pccp')).toBe(0);
  });

  it('returns 0 when the requested field is missing', () => {
    const bare: CountyData = { county: 'X', region: 'Y', count: 10, fips: '' };
    expect(valueForRate(bare, 'numerator')).toBe(0);
    expect(valueForRate(bare, 'denominator')).toBe(0);
  });
});

describe('useFilteredData — getCountRangeForRate', () => {
  const data: CountyData[] = [
    { county: 'A', region: 'R', count: 10, fips: '', casePatients: 4, totalPatients: 40 },
    { county: 'B', region: 'R', count: 50, fips: '', casePatients: 45, totalPatients: 90 },
  ];

  it('pccp mode ranges over the count field', () => {
    expect(getCountRangeForRate(data, 'pccp')).toEqual({ min: 10, max: 50 });
  });

  it('numerator mode ranges over casePatients', () => {
    expect(getCountRangeForRate(data, 'numerator')).toEqual({ min: 4, max: 45 });
  });

  it('denominator mode ranges over totalPatients', () => {
    expect(getCountRangeForRate(data, 'denominator')).toEqual({ min: 40, max: 90 });
  });

  it('falls back to {min:0, max:1} when there is no positive data', () => {
    expect(getCountRangeForRate([], 'pccp')).toEqual({ min: 0, max: 1 });
  });
});
