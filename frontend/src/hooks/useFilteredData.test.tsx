import { renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MOCK_COUNTY_DATA } from '../data/mockData';
import type { FilterState } from '../types';
import {
  applyFilters,
  getCountRange,
  useCountyDataMap,
  useFilteredData,
} from './useFilteredData';

const defaultFilters: FilterState = {
  rateType: 'incidence',
  sex: 'all',
  cancerType: 'All Types',
  breed: 'All Breeds',
};

describe('useFilteredData', () => {
  it('returns full mock county data for default filters', async () => {
    const { result } = renderHook(() => useFilteredData(defaultFilters));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.countyData).toEqual(MOCK_COUNTY_DATA);
  });

  it('returns deterministic filtered county data for the same filters', () => {
    const filters: FilterState = {
      ...defaultFilters,
      sex: 'female_spayed',
      cancerType: 'Lymphoma',
      breed: 'Golden Retriever',
    };

    expect(applyFilters(MOCK_COUNTY_DATA, filters)).toEqual(applyFilters(MOCK_COUNTY_DATA, filters));
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
