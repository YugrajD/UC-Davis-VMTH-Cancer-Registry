import { useEffect, useState } from 'react';
import type { FilterState } from '../types';
import type { IncidenceRecord } from '../api/client';
import { MOCK_CANCER_TYPE_INCIDENTS } from '../data/mockData';

// Deterministic seeded random for stable filter results
function seededRandom(seed: string) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(31, h) + seed.charCodeAt(i) | 0;
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
    h = Math.imul(h ^ (h >>> 13), 0x45d9f3b);
    h = (h ^ (h >>> 16)) >>> 0;
    return (h % 100) / 100;
  };
}

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
    setLoading(true);
    setError(null);

    let result = MOCK_CANCER_TYPE_INCIDENTS;

    // If a specific cancer type is selected, show only that one
    if (filters.cancerType && filters.cancerType !== 'All Types') {
      const match = result.find(r =>
        r.cancer_type.toLowerCase().includes(filters.cancerType.toLowerCase())
      );
      result = match ? [match] : result;
    }

    // If sex filter is applied, scale counts down with per-type variation
    if (filters.sex && filters.sex !== 'all') {
      const rand = seededRandom(filters.sex);
      result = result.map(r => ({
        ...r,
        count: Math.max(1, Math.round(r.count * 0.25 * (0.6 + rand() * 0.8))),
      }));
    }

    // If breed filter is applied, scale counts with breed-specific variation
    if (filters.breed && filters.breed !== 'All Breeds') {
      const rand = seededRandom(filters.breed);
      result = result.map(r => ({
        ...r,
        count: Math.max(1, Math.round(r.count * 0.1 * (0.5 + rand() * 1.0))),
      }));
    }

    setData(result);
    setLoading(false);
  }, [filters.cancerType, filters.sex, filters.breed]);

  return { data, loading, error };
}
