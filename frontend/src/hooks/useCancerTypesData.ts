import { useEffect, useMemo, useState } from 'react';
import type { FilterState } from '../types';
import { fetchIncidenceByCancerType, type IncidenceRecord } from '../api/client';

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

export interface CancerTypesState {
  data: IncidenceRecord[];
  loading: boolean;
  error: string | null;
}

export function applyCancerTypeDemoFilters(
  data: IncidenceRecord[],
  filters: FilterState,
  options: { applyServerSideFilters?: boolean } = {},
): IncidenceRecord[] {
  let result = data;

  if (options.applyServerSideFilters !== false) {
    if (filters.cancerType && filters.cancerType !== 'All Types') {
      const match = result.find(r =>
        r.cancer_type.toLowerCase().includes(filters.cancerType.toLowerCase())
      );
      result = match ? [match] : result;
    }

    if (filters.sex && filters.sex !== 'all') {
      const rand = seededRandom(filters.sex);
      result = result.map(r => ({
        ...r,
        count: Math.max(1, Math.round(r.count * 0.25 * (0.6 + rand() * 0.8))),
      }));
    }
  }

  // Breed-level cancer-type aggregation is not available from the current
  // API, so preserve the previous deterministic demo narrowing for breed.
  if (filters.breed && filters.breed !== 'All Breeds') {
    const rand = seededRandom(filters.breed);
    result = result.map(r => ({
      ...r,
      count: Math.max(1, Math.round(r.count * 0.1 * (0.5 + rand() * 1.0))),
    }));
  }

  return result;
}

export function useCancerTypesData(filters: FilterState): CancerTypesState {
  const [apiData, setApiData] = useState<IncidenceRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadCancerTypes() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetchIncidenceByCancerType({
          cancerTypes:
            filters.cancerType && filters.cancerType !== 'All Types'
              ? [filters.cancerType]
              : undefined,
          sex: filters.sex && filters.sex !== 'all' ? filters.sex : undefined,
        });
        if (!cancelled) {
          setApiData(response.data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unable to load cancer type data');
          setApiData([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadCancerTypes();

    return () => {
      cancelled = true;
    };
  }, [filters.cancerType, filters.sex]);

  const data = useMemo(() => {
    return applyCancerTypeDemoFilters(apiData, filters, { applyServerSideFilters: false });
  }, [apiData, filters]);

  return { data, loading, error };
}
