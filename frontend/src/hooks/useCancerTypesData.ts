import { useEffect, useState } from 'react';
import type { FilterState } from '../types';
import { fetchIncidenceByCancerType, type IncidenceRecord } from '../api/client';

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
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const apiFilters: { cancerTypes?: string[]; sex?: string } = {};
        if (filters.cancerType && filters.cancerType !== 'All Types') {
          apiFilters.cancerTypes = [filters.cancerType];
        }
        if (filters.sex && filters.sex !== 'all') {
          apiFilters.sex = filters.sex;
        }
        const res = await fetchIncidenceByCancerType(apiFilters);
        setData(res.data);
      } catch (err) {
        console.error('Failed to load cancer types incidence:', err);
        setError(err instanceof Error ? err.message : 'Failed to load cancer type data');
        setData([]);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [filters.cancerType, filters.sex]);

  return { data, loading, error };
}

