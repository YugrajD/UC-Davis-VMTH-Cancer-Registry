/**
 * Helpers for shaping the /api/v1/trends/* response into chart-ready data.
 *
 * The backend returns one series per cancer type.  At ~30 cancer types the
 * multi-line chart is unreadable, so we keep the top N by total case count
 * and aggregate the remainder into a single "Other" line that the user can
 * toggle on or off.
 */

export interface TrendPoint {
  year: number;
  count: number;
  pccp?: number;
  total_patients?: number;
}

export interface TrendSeries {
  name: string;
  data: TrendPoint[];
}

export const OTHER_SERIES_NAME = 'Other';

/**
 * Reduce a list of series to the top `n` by total count, plus one combined
 * "Other" series aggregating the remainder by year.  When `series.length <= n`
 * the original input is returned (no "Other" appended).  Pure function.
 */
export function topNWithOther(series: TrendSeries[], n: number): TrendSeries[] {
  if (n <= 0) return [];
  if (series.length <= n) {
    return series.map((s) => ({ name: s.name, data: [...s.data] }));
  }

  const totals = new Map<string, number>();
  for (const s of series) {
    let t = 0;
    for (const p of s.data) t += p.count;
    totals.set(s.name, t);
  }

  const sorted = [...series].sort(
    (a, b) => (totals.get(b.name) ?? 0) - (totals.get(a.name) ?? 0),
  );
  const top = sorted.slice(0, n);
  const rest = sorted.slice(n);

  // For Other: sum patient counts per year; total_patients is shared across all series in a year.
  const otherByYear = new Map<number, { count: number; total_patients?: number }>();
  for (const s of rest) {
    for (const p of s.data) {
      const cur = otherByYear.get(p.year) ?? { count: 0, total_patients: p.total_patients };
      otherByYear.set(p.year, {
        count: cur.count + p.count,
        total_patients: cur.total_patients ?? p.total_patients,
      });
    }
  }
  const otherData: TrendPoint[] = Array.from(otherByYear.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([year, { count, total_patients }]) => {
      const point: TrendPoint = { year, count };
      if (total_patients != null && total_patients > 0) {
        point.total_patients = total_patients;
        point.pccp = Math.round((count / total_patients) * 100 * 100) / 100;
      }
      return point;
    });

  return [
    ...top.map((s) => ({ name: s.name, data: [...s.data] })),
    { name: OTHER_SERIES_NAME, data: otherData },
  ];
}

/**
 * Compute the union of all years across every series (sorted ascending).
 * Used to set the chart's x-axis domain so years with no data still appear.
 */
export function yearRange(series: TrendSeries[]): number[] {
  const years = new Set<number>();
  for (const s of series) {
    for (const p of s.data) years.add(p.year);
  }
  return Array.from(years).sort((a, b) => a - b);
}

/**
 * Look up a series' count for a given year, returning 0 when the year is
 * absent from the series data (rather than skipping the point — keeps the
 * line continuous across years where one cancer type wasn't diagnosed).
 */
export function countForYear(series: TrendSeries, year: number): number {
  for (const p of series.data) {
    if (p.year === year) return p.count;
  }
  return 0;
}

/**
 * Look up a series' PCCP for a given year, returning null when absent.
 * Returns null (not 0) so callers can distinguish "no data" from "zero rate".
 */
export function pccpForYear(series: TrendSeries, year: number): number | null {
  for (const p of series.data) {
    if (p.year === year) return p.pccp ?? null;
  }
  return null;
}
