import { useEffect, useState } from 'react';
import type { FilterState } from '../types';
import type { IncidenceRecord } from '../api/client';
import { MOCK_CANCER_TYPE_INCIDENTS } from '../data/mockData';

interface CancerTypesState {
  data: IncidenceRecord[];
  loading: boolean;
  error: string | null;
}

export function useCancerTypesData(filters: FilterState): CancerTypesState {
  const [data, setData] = useState<IncidenceRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // For the presentation, use the static cancer-type distribution
    // so the bar chart always matches the screenshot.
    setLoading(true);
    setError(null);
    setData(MOCK_CANCER_TYPE_INCIDENTS);
    setLoading(false);
  }, [filters.cancerType, filters.sex]);

  return { data, loading, error };
}

