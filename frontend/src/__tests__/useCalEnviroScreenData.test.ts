import { describe, it, expect } from 'vitest';
import { MOCK_CALENVIROSCREEN_DATA } from '../data/mockData';

describe('MOCK_CALENVIROSCREEN_DATA structure', () => {
  it('dataset covers all 58 California counties', () => {
    expect(MOCK_CALENVIROSCREEN_DATA.length).toBe(58);
  });

  it('every entry has a county_name string', () => {
    for (const entry of MOCK_CALENVIROSCREEN_DATA) {
      expect(typeof entry.county_name).toBe('string');
      expect(entry.county_name.length).toBeGreaterThan(0);
    }
  });

  it('every entry has a county_fips code', () => {
    for (const entry of MOCK_CALENVIROSCREEN_DATA) {
      expect(typeof entry.county_fips).toBe('string');
      expect(entry.county_fips.length).toBeGreaterThan(0);
    }
  });

  it('FIPS codes start with 06 (California)', () => {
    for (const entry of MOCK_CALENVIROSCREEN_DATA) {
      expect(entry.county_fips.startsWith('06')).toBe(true);
    }
  });

  it('ces_score is a number or null (no undefined)', () => {
    for (const entry of MOCK_CALENVIROSCREEN_DATA) {
      expect(entry.ces_score === null || typeof entry.ces_score === 'number').toBe(true);
    }
  });

  it('numeric indicator values are within 0–100 percentile range', () => {
    for (const entry of MOCK_CALENVIROSCREEN_DATA) {
      if (entry.ces_score !== null) {
        expect(entry.ces_score).toBeGreaterThanOrEqual(0);
        expect(entry.ces_score).toBeLessThanOrEqual(100);
      }
    }
  });

  it('county names are unique', () => {
    const names = MOCK_CALENVIROSCREEN_DATA.map(d => d.county_name);
    const uniqueNames = new Set(names);
    expect(uniqueNames.size).toBe(names.length);
  });

  it('includes Sacramento county', () => {
    const sac = MOCK_CALENVIROSCREEN_DATA.find(d => d.county_name === 'Sacramento');
    expect(sac).toBeDefined();
  });
});
