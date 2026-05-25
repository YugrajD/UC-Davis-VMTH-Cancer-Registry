import { describe, it, expect } from 'vitest';
import {
  topNWithOther,
  yearRange,
  countForYear,
  OTHER_SERIES_NAME,
  type TrendSeries,
} from '../lib/trends';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const lymphoma: TrendSeries = {
  name: 'Lymphoma',
  data: [
    { year: 2020, count: 50 },
    { year: 2021, count: 60 },
    { year: 2022, count: 70 },
  ],
};
const mastCell: TrendSeries = {
  name: 'Mast Cell Tumor',
  data: [
    { year: 2020, count: 30 },
    { year: 2021, count: 32 },
    { year: 2022, count: 35 },
  ],
};
const osteo: TrendSeries = {
  name: 'Osteosarcoma',
  data: [
    { year: 2020, count: 10 },
    { year: 2021, count: 12 },
  ],
};
const rare1: TrendSeries = {
  name: 'Hemangiosarcoma',
  data: [{ year: 2022, count: 4 }],
};
const rare2: TrendSeries = {
  name: 'Melanoma',
  data: [{ year: 2021, count: 3 }],
};

// ---------------------------------------------------------------------------
// topNWithOther
// ---------------------------------------------------------------------------

describe('topNWithOther', () => {
  it('returns input as-is when there are fewer series than the limit', () => {
    const result = topNWithOther([lymphoma, mastCell], 5);
    expect(result).toHaveLength(2);
    expect(result.map((s) => s.name)).toEqual(['Lymphoma', 'Mast Cell Tumor']);
    // Verify no "Other" line was appended.
    expect(result.find((s) => s.name === OTHER_SERIES_NAME)).toBeUndefined();
  });

  it('returns input as-is when exactly at the limit', () => {
    const result = topNWithOther([lymphoma, mastCell, osteo], 3);
    expect(result).toHaveLength(3);
    expect(result.find((s) => s.name === OTHER_SERIES_NAME)).toBeUndefined();
  });

  it('keeps top-N by total count and folds remainder into "Other"', () => {
    const result = topNWithOther([lymphoma, mastCell, osteo, rare1, rare2], 3);
    expect(result).toHaveLength(4); // 3 top + Other
    expect(result.slice(0, 3).map((s) => s.name)).toEqual([
      'Lymphoma',         // total 180
      'Mast Cell Tumor',  // total 97
      'Osteosarcoma',     // total 22
    ]);
    expect(result[3].name).toBe(OTHER_SERIES_NAME);
  });

  it('aggregates "Other" correctly by year', () => {
    const result = topNWithOther([lymphoma, mastCell, osteo, rare1, rare2], 3);
    const other = result.find((s) => s.name === OTHER_SERIES_NAME)!;
    // rare1 contributes 4 to 2022, rare2 contributes 3 to 2021.
    expect(other.data).toEqual([
      { year: 2021, count: 3 },
      { year: 2022, count: 4 },
    ]);
  });

  it('preserves the per-series year data for top series', () => {
    const result = topNWithOther([lymphoma, mastCell, osteo, rare1, rare2], 1);
    expect(result[0].data).toEqual(lymphoma.data);
    // Ensure deep copy — caller mutating the output should not affect the input.
    result[0].data.push({ year: 9999, count: 999 });
    expect(lymphoma.data).not.toContainEqual({ year: 9999, count: 999 });
  });

  it('returns an empty array when n is 0', () => {
    const result = topNWithOther([lymphoma, mastCell], 0);
    expect(result).toEqual([]);
  });

  it('handles an empty input list', () => {
    expect(topNWithOther([], 5)).toEqual([]);
  });

  it('breaks ties deterministically by input order', () => {
    const a: TrendSeries = { name: 'A', data: [{ year: 2020, count: 10 }] };
    const b: TrendSeries = { name: 'B', data: [{ year: 2020, count: 10 }] };
    const c: TrendSeries = { name: 'C', data: [{ year: 2020, count: 10 }] };
    const result = topNWithOther([a, b, c], 2);
    // Stable sort: A and B come first; C lands in Other.
    expect(result.slice(0, 2).map((s) => s.name)).toEqual(['A', 'B']);
    expect(result[2].name).toBe(OTHER_SERIES_NAME);
  });
});

// ---------------------------------------------------------------------------
// yearRange
// ---------------------------------------------------------------------------

describe('yearRange', () => {
  it('returns the union of years across all series, sorted ascending', () => {
    const result = yearRange([lymphoma, osteo, rare1]);
    expect(result).toEqual([2020, 2021, 2022]);
  });

  it('returns an empty array for an empty input', () => {
    expect(yearRange([])).toEqual([]);
  });

  it('deduplicates years that appear in multiple series', () => {
    const result = yearRange([lymphoma, mastCell, osteo]);
    expect(result).toEqual([2020, 2021, 2022]);
  });

  it('handles disjoint year ranges', () => {
    const old: TrendSeries = {
      name: 'old',
      data: [{ year: 2010, count: 1 }],
    };
    const recent: TrendSeries = {
      name: 'recent',
      data: [{ year: 2025, count: 1 }],
    };
    expect(yearRange([old, recent])).toEqual([2010, 2025]);
  });
});

// ---------------------------------------------------------------------------
// countForYear
// ---------------------------------------------------------------------------

describe('countForYear', () => {
  it('returns the count for the matching year', () => {
    expect(countForYear(lymphoma, 2021)).toBe(60);
  });

  it('returns 0 when the year is absent — keeps the line continuous', () => {
    expect(countForYear(osteo, 2022)).toBe(0);
  });

  it('returns 0 for an empty series', () => {
    expect(countForYear({ name: 'x', data: [] }, 2020)).toBe(0);
  });
});
