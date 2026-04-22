import { describe, it, expect } from 'vitest';
import { useCalEnviroScreenData } from '../hooks/useCalEnviroScreenData';
import { MOCK_CALENVIROSCREEN_DATA } from '../data/mockData';

describe('useCalEnviroScreenData', () => {
  it('returns loading: false (synchronous mock data)', () => {
    expect(useCalEnviroScreenData().loading).toBe(false);
  });

  it('returns error: null', () => {
    expect(useCalEnviroScreenData().error).toBeNull();
  });

  it('returns the full CalEnviroScreen mock dataset', () => {
    const { data } = useCalEnviroScreenData();
    expect(data).toBe(MOCK_CALENVIROSCREEN_DATA);
  });

  it('dataset covers all 58 California counties', () => {
    const { data } = useCalEnviroScreenData();
    expect(data.length).toBe(58);
  });

  it('every entry has a county_name string', () => {
    const { data } = useCalEnviroScreenData();
    for (const entry of data) {
      expect(typeof entry.county_name).toBe('string');
      expect(entry.county_name.length).toBeGreaterThan(0);
    }
  });

  it('every entry has a county_fips code', () => {
    const { data } = useCalEnviroScreenData();
    for (const entry of data) {
      expect(typeof entry.county_fips).toBe('string');
      expect(entry.county_fips.length).toBeGreaterThan(0);
    }
  });

  it('FIPS codes start with 06 (California)', () => {
    const { data } = useCalEnviroScreenData();
    for (const entry of data) {
      expect(entry.county_fips.startsWith('06')).toBe(true);
    }
  });

  it('ces_score is a number or null (no undefined)', () => {
    const { data } = useCalEnviroScreenData();
    for (const entry of data) {
      expect(entry.ces_score === null || typeof entry.ces_score === 'number').toBe(true);
    }
  });

  it('numeric indicator values are within 0–100 percentile range', () => {
    const { data } = useCalEnviroScreenData();
    for (const entry of data) {
      if (entry.ces_score !== null) {
        expect(entry.ces_score).toBeGreaterThanOrEqual(0);
        expect(entry.ces_score).toBeLessThanOrEqual(100);
      }
    }
  });

  it('county names are unique', () => {
    const { data } = useCalEnviroScreenData();
    const names = data.map(d => d.county_name);
    const uniqueNames = new Set(names);
    expect(uniqueNames.size).toBe(names.length);
  });

  it('includes Sacramento county', () => {
    const { data } = useCalEnviroScreenData();
    const sac = data.find(d => d.county_name === 'Sacramento');
    expect(sac).toBeDefined();
  });

  it('is idempotent — returns same reference each call', () => {
    expect(useCalEnviroScreenData().data).toBe(useCalEnviroScreenData().data);
  });
});
