import { useEffect, useState } from 'react';
import type { CalEnviroScreenData } from '../types';
import { fetchCalEnviroScreen } from '../api/client';

export function useCalEnviroScreenData(): { data: CalEnviroScreenData[]; loading: boolean; error: string | null } {
  const [data, setData] = useState<CalEnviroScreenData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCalEnviroScreen()
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Failed to load'))
      .finally(() => setLoading(false));
  }, []);

  return { data, loading, error };
}
