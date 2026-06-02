import { renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { MOCK_CANCER_TYPE_INCIDENTS } from '../data/mockData';
import type { FilterState } from '../types';
import { useCancerTypesData } from './useCancerTypesData';

const mockFetchIncidenceByCancerType = vi.hoisted(() => vi.fn());

vi.mock('../api/client', () => ({
  fetchIncidenceByCancerType: mockFetchIncidenceByCancerType,
}));

const defaultFilters: FilterState = {
  rateType: 'incidence',
  sex: 'all',
  ageGroup: 'all',
  cancerType: 'All Types',
  breed: 'All Breeds',
};

describe('useCancerTypesData', () => {
  it('returns all mock cancer types by default', async () => {
    mockFetchIncidenceByCancerType.mockResolvedValueOnce({
      data: MOCK_CANCER_TYPE_INCIDENTS,
      total: MOCK_CANCER_TYPE_INCIDENTS.length,
      filters_applied: {},
    });

    const { result } = renderHook(() => useCancerTypesData(defaultFilters));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toEqual(MOCK_CANCER_TYPE_INCIDENTS);
  });

  it('narrows to the selected cancer type', async () => {
    const lymphomaOnly = [{ cancer_type: 'Lymphoma', count: 1250 }];
    mockFetchIncidenceByCancerType.mockResolvedValueOnce({
      data: lymphomaOnly,
      total: 1,
      filters_applied: { cancer_type: ['Lymphoma'] },
    });

    const { result } = renderHook(() => useCancerTypesData({
      ...defaultFilters,
      cancerType: 'Lymphoma',
    }));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toEqual(lymphomaOnly);
  });

  it('deterministically reduces sex and breed counts without dropping below 1', async () => {
    const filters: FilterState = {
      ...defaultFilters,
      sex: 'male_neutered',
      breed: 'Golden Retriever',
    };
    const reducedData = MOCK_CANCER_TYPE_INCIDENTS.map(r => ({
      ...r,
      count: Math.max(1, Math.floor(r.count * 0.4)),
    }));
    mockFetchIncidenceByCancerType.mockResolvedValue({
      data: reducedData,
      total: reducedData.length,
      filters_applied: {},
    });

    const first = renderHook(() => useCancerTypesData(filters));
    const second = renderHook(() => useCancerTypesData(filters));

    await waitFor(() => expect(first.result.current.loading).toBe(false));
    await waitFor(() => expect(second.result.current.loading).toBe(false));

    expect(first.result.current.data).toEqual(second.result.current.data);
    expect(first.result.current.data.every(record => record.count >= 1)).toBe(true);
    expect(first.result.current.data[0].count).toBeLessThan(MOCK_CANCER_TYPE_INCIDENTS[0].count);
  });
});
