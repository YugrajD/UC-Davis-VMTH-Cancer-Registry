import { useEffect, useState } from 'react';
import { fetchTrendsByCancerType } from '../api/client';
import { topNWithOther, type TrendSeries } from '../lib/trends';

export interface YearlyTrendsState {
  series: TrendSeries[];
  loading: boolean;
  error: string | null;
}

const TOP_N = 5;

/**
 * Fetches /api/v1/trends/by-cancer-type once on mount and reduces the
 * full per-cancer-type series list to the top N by total case count plus
 * an aggregated "Other" line.  Mirrors the useCancerTypesData pattern:
 * async fetch in useEffect with a cancelled flag to handle unmount.
 */
export function useYearlyTrendsData(): YearlyTrendsState {
  const [series, setSeries] = useState<TrendSeries[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await fetchTrendsByCancerType();
        if (cancelled) return;
        // Backend's TrendPointApi includes deceased/alive — drop them; the
        // chart only renders total counts per year.
        const normalized: TrendSeries[] = response.series.map((s) => ({
          name: s.name,
          data: s.data.map((p) => ({ year: p.year, count: p.count })),
        }));
        setSeries(topNWithOther(normalized, TOP_N));
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unable to load trend data');
          setSeries([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  return { series, loading, error };
}
