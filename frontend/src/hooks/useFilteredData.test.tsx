import { renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { MOCK_COUNTY_DATA } from '../data/mockData';
import type { FilterState } from '../types';
import {
  applyCountyDemoFilters,
  getCountRange,
  useCountyDataMap,
  useFilteredData,
} from './useFilteredData';

vi.mock('../api/client', () => ({
  fetchPCCPByCounty: vi.fn().mockResolvedValue({
    data: MOCK_COUNTY_DATA.map(c => ({
      county: c.county,
      cancer_patients: c.count,
      total_patients: c.count * 10,
      pccp: c.count / (c.count * 10) * 100,
    })),
    overall_cancer_patients: MOCK_COUNTY_DATA.reduce((s, c) => s + c.count, 0),
    overall_total_patients: MOCK_COUNTY_DATA.reduce((s, c) => s + c.count * 10, 0),
    overall_pccp: 10,
  }),
}));

const defaultFilters: FilterState = {
  rateType: 'incidence',
  sex: 'all',
  ageGroup: 'all',
  cancerType: 'All Types',
  breed: 'All Breeds',
};

describe('useFilteredData', () => {
  it('returns full mock county data for default filters', async () => {
    const { result } = renderHook(() => useFilteredData(defaultFilters));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.countyData).toHaveLength(MOCK_COUNTY_DATA.length);
    expect(result.current.countyData.map(c => c.county)).toEqual(MOCK_COUNTY_DATA.map(c => c.county));
  });

  it('returns deterministic filtered county data for the same filters', () => {
    const filters: FilterState = {
      ...defaultFilters,
      sex: 'female_spayed',
      cancerType: 'Lymphoma',
      breed: 'Golden Retriever',
    };

    expect(applyCountyDemoFilters(MOCK_COUNTY_DATA, filters)).toEqual(applyCountyDemoFilters(MOCK_COUNTY_DATA, filters));
  });

  it('returns a default count range for empty or zero county counts', () => {
    expect(getCountRange([])).toEqual({ min: 0, max: 1 });
    expect(getCountRange([{ county: 'Zero', region: 'Nowhere', count: 0, fips: '00000' }])).toEqual({ min: 0, max: 1 });
  });

  it('generates a region summary with California root and UC Davis catchment area', async () => {
    const { result } = renderHook(() => useFilteredData(defaultFilters));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.regionSummary.name).toBe('California');
    expect(result.current.regionSummary.children?.[0].name).toBe('UC Davis Catchment Area');
  });

  it('maps county names to county records', () => {
    const { result } = renderHook(() => useCountyDataMap(MOCK_COUNTY_DATA));

    expect(result.current.get('Sacramento')).toEqual(MOCK_COUNTY_DATA[0]);
    expect(result.current.get('Yolo')?.county).toBe('Yolo');
  });
});
