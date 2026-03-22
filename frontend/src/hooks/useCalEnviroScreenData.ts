import { useEffect, useState } from 'react';
import type { CalEnviroScreenData } from '../types';
import { MOCK_CALENVIROSCREEN_DATA } from '../data/mockData';

export function useCalEnviroScreenData() {
  const [data, setData] = useState<CalEnviroScreenData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setData(MOCK_CALENVIROSCREEN_DATA);
    setLoading(false);
  }, []);

  return { data, loading, error };
}
