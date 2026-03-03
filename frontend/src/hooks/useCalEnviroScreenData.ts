import { useEffect, useState } from 'react';
import type { CalEnviroScreenData } from '../types';
import { fetchCalEnviroScreen } from '../api/client';

export function useCalEnviroScreenData() {
  const [data, setData] = useState<CalEnviroScreenData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchCalEnviroScreen();
        if (!cancelled) setData(result);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load CalEnviroScreen data');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, []);

  return { data, loading, error };
}
