import { describe, it, expect } from 'vitest';
import {
  MOCK_BREEDS,
  MOCK_COUNTY_DATA,
  MOCK_CANCER_TYPE_INCIDENTS,
  MOCK_CALENVIROSCREEN_DATA,
  getMockBreedDetail,
} from '../data/mockData';

// ---------------------------------------------------------------------------
// MOCK_BREEDS
// ---------------------------------------------------------------------------

describe('MOCK_BREEDS', () => {
  it('contains 10 breeds', () => {
    expect(MOCK_BREEDS.length).toBe(10);
  });

  it('includes Golden Retriever', () => {
    expect(MOCK_BREEDS).toContain('Golden Retriever');
  });

  it('includes Labrador Retriever', () => {
    expect(MOCK_BREEDS).toContain('Labrador Retriever');
  });

  it('all entries are non-empty strings', () => {
    for (const breed of MOCK_BREEDS) {
      expect(typeof breed).toBe('string');
      expect(breed.length).toBeGreaterThan(0);
    }
  });

  it('breed names are unique', () => {
    const unique = new Set(MOCK_BREEDS);
    expect(unique.size).toBe(MOCK_BREEDS.length);
  });
});

// ---------------------------------------------------------------------------
// MOCK_COUNTY_DATA
// ---------------------------------------------------------------------------

describe('MOCK_COUNTY_DATA', () => {
  it('contains county entries', () => {
    expect(MOCK_COUNTY_DATA.length).toBeGreaterThan(0);
  });

  it('includes Sacramento', () => {
    expect(MOCK_COUNTY_DATA.some(c => c.county === 'Sacramento')).toBe(true);
  });

  it('every county has a positive count', () => {
    for (const c of MOCK_COUNTY_DATA) {
      expect(c.count).toBeGreaterThan(0);
    }
  });

  it('every county has a region', () => {
    for (const c of MOCK_COUNTY_DATA) {
      expect(typeof c.region).toBe('string');
      expect(c.region.length).toBeGreaterThan(0);
    }
  });

  it('every county has a fips field (may be empty string pending data backfill)', () => {
    for (const c of MOCK_COUNTY_DATA) {
      expect(typeof c.fips).toBe('string');
    }
  });

  it('county names are unique', () => {
    const names = MOCK_COUNTY_DATA.map(c => c.county);
    expect(new Set(names).size).toBe(names.length);
  });

  it('covers the dashboard catchment regions represented in mock data', () => {
    const regions = new Set(MOCK_COUNTY_DATA.map(c => c.region));
    expect(regions.has('San Francisco Bay Area')).toBe(true);
    expect(regions.has('Sacramento Valley')).toBe(true);
    expect(regions.has('San Joaquin Valley')).toBe(true);
    expect(regions.has('Sierra Nevada')).toBe(true);
  });

  it('Sacramento has the highest case count in the dataset', () => {
    const sac = MOCK_COUNTY_DATA.find(c => c.county === 'Sacramento')!;
    const max = Math.max(...MOCK_COUNTY_DATA.map(c => c.count));
    expect(sac.count).toBe(max);
  });
});

// ---------------------------------------------------------------------------
// MOCK_CANCER_TYPE_INCIDENTS
// ---------------------------------------------------------------------------

describe('MOCK_CANCER_TYPE_INCIDENTS', () => {
  it('contains 8 cancer type records', () => {
    expect(MOCK_CANCER_TYPE_INCIDENTS.length).toBe(8);
  });

  it('includes Lymphoma', () => {
    expect(MOCK_CANCER_TYPE_INCIDENTS.some(r => r.cancer_type === 'Lymphoma')).toBe(true);
  });

  it('all records have positive counts', () => {
    for (const r of MOCK_CANCER_TYPE_INCIDENTS) {
      expect(r.count).toBeGreaterThan(0);
    }
  });

  it('cancer type names are unique', () => {
    const names = MOCK_CANCER_TYPE_INCIDENTS.map(r => r.cancer_type);
    expect(new Set(names).size).toBe(names.length);
  });

  it('total cases across all types is approximately 5000', () => {
    const total = MOCK_CANCER_TYPE_INCIDENTS.reduce((s, r) => s + r.count, 0);
    expect(total).toBeGreaterThan(4000);
    expect(total).toBeLessThan(6000);
  });

  it('Lymphoma is the most common cancer type', () => {
    const lymphoma = MOCK_CANCER_TYPE_INCIDENTS.find(r => r.cancer_type === 'Lymphoma')!;
    const max = Math.max(...MOCK_CANCER_TYPE_INCIDENTS.map(r => r.count));
    expect(lymphoma.count).toBe(max);
  });
});

// ---------------------------------------------------------------------------
// getMockBreedDetail
// ---------------------------------------------------------------------------

describe('getMockBreedDetail', () => {
  it('returns a BreedDetail with the correct breed name', () => {
    const detail = getMockBreedDetail('Golden Retriever');
    expect(detail.breed).toBe('Golden Retriever');
  });

  it('total_cases is a positive number', () => {
    const detail = getMockBreedDetail('Golden Retriever');
    expect(detail.total_cases).toBeGreaterThan(0);
  });

  it('sex_breakdown has four entries (Male, Female, Neutered Male, Spayed Female)', () => {
    const detail = getMockBreedDetail('Golden Retriever');
    expect(detail.sex_breakdown.length).toBe(4);
    const sexLabels = detail.sex_breakdown.map(s => s.sex);
    expect(sexLabels).toContain('Male');
    expect(sexLabels).toContain('Female');
    expect(sexLabels).toContain('Neutered Male');
    expect(sexLabels).toContain('Spayed Female');
  });

  it('sex_breakdown counts are all positive', () => {
    const detail = getMockBreedDetail('Golden Retriever');
    for (const s of detail.sex_breakdown) {
      expect(s.count).toBeGreaterThan(0);
    }
  });

  it('sex_breakdown counts sum to total_cases', () => {
    const detail = getMockBreedDetail('Golden Retriever');
    const sexTotal = detail.sex_breakdown.reduce((s, x) => s + x.count, 0);
    expect(sexTotal).toBe(detail.total_cases);
  });

  it('cancer_types array is non-empty', () => {
    const detail = getMockBreedDetail('Golden Retriever');
    expect(detail.cancer_types.length).toBeGreaterThan(0);
  });

  it('each cancer type has a name and positive count', () => {
    const detail = getMockBreedDetail('Golden Retriever');
    for (const ct of detail.cancer_types) {
      expect(typeof ct.cancer_type).toBe('string');
      expect(ct.count).toBeGreaterThan(0);
    }
  });

  it('county_cases array is non-empty', () => {
    const detail = getMockBreedDetail('Golden Retriever');
    expect(detail.county_cases.length).toBeGreaterThan(0);
  });

  it('each county case has a county_name, fips_code, and positive count', () => {
    const detail = getMockBreedDetail('Golden Retriever');
    for (const cc of detail.county_cases) {
      expect(typeof cc.county_name).toBe('string');
      expect(cc.fips_code.startsWith('06')).toBe(true);
      expect(cc.count).toBeGreaterThan(0);
    }
  });

  it('is deterministic — same breed always returns same data', () => {
    const a = getMockBreedDetail('Boxer');
    const b = getMockBreedDetail('Boxer');
    expect(a.total_cases).toBe(b.total_cases);
    expect(a.sex_breakdown.map(s => s.count)).toEqual(b.sex_breakdown.map(s => s.count));
  });

  it('all MOCK_BREEDS have details with valid total_cases', () => {
    for (const breed of MOCK_BREEDS) {
      const detail = getMockBreedDetail(breed);
      expect(detail.breed).toBe(breed);
      expect(detail.total_cases).toBeGreaterThan(0);
    }
  });

  it('different breeds return different case counts', () => {
    const golden = getMockBreedDetail('Golden Retriever');
    const labrador = getMockBreedDetail('Labrador Retriever');
    // They may have different distributions even if totals are close
    const goldenCancerCounts = golden.cancer_types.map(c => c.count);
    const labradorCancerCounts = labrador.cancer_types.map(c => c.count);
    expect(goldenCancerCounts).not.toEqual(labradorCancerCounts);
  });
});

// ---------------------------------------------------------------------------
// MOCK_CALENVIROSCREEN_DATA
// ---------------------------------------------------------------------------

describe('MOCK_CALENVIROSCREEN_DATA', () => {
  it('covers all 58 California counties', () => {
    expect(MOCK_CALENVIROSCREEN_DATA.length).toBe(58);
  });

  it('each entry has a county_id', () => {
    for (const entry of MOCK_CALENVIROSCREEN_DATA) {
      expect(typeof entry.county_id).toBe('number');
    }
  });

  it('each entry has a county_name string', () => {
    for (const entry of MOCK_CALENVIROSCREEN_DATA) {
      expect(typeof entry.county_name).toBe('string');
      expect(entry.county_name.length).toBeGreaterThan(0);
    }
  });

  it('FIPS codes are unique and start with 06', () => {
    const fipsCodes = MOCK_CALENVIROSCREEN_DATA.map(d => d.county_fips);
    for (const fips of fipsCodes) {
      expect(fips.startsWith('06')).toBe(true);
    }
    expect(new Set(fipsCodes).size).toBe(fipsCodes.length);
  });

  it('numeric indicator values are null or in 0–100 range', () => {
    for (const entry of MOCK_CALENVIROSCREEN_DATA) {
      if (entry.ces_score !== null) {
        expect(entry.ces_score).toBeGreaterThanOrEqual(0);
        expect(entry.ces_score).toBeLessThanOrEqual(100);
      }
      if (entry.ozone !== null) {
        expect(entry.ozone).toBeGreaterThanOrEqual(0);
        expect(entry.ozone).toBeLessThanOrEqual(100);
      }
    }
  });
});
