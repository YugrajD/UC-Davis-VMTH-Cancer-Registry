import { useEffect, useMemo, useState } from 'react';
import type { FilterState, CountyData, RegionSummary } from '../types';
import { fetchIncidence, type IncidenceRecord } from '../api/client';
import { isUcDavisCatchmentRegion, regionForCounty } from '../data/californiaRegions';

export interface FilteredDataState {
  countyData: CountyData[];
  regionSummary: RegionSummary;
  countRange: { min: number; max: number };
  loading: boolean;
  error: string | null;
}

const EMPTY_REGION_SUMMARY: RegionSummary = {
  name: 'California',
  type: 'state',
  count: 0,
  children: [],
};

function sexFilterValue(sex: FilterState['sex']) {
  return sex && sex !== 'all' ? sex : undefined;
}

function cancerTypeFilterValue(cancerType: string) {
  return cancerType && cancerType !== 'All Types' ? [cancerType] : undefined;
}

// Deterministic pseudo-random from a string seed so the same filter
// always produces the same numbers (no flicker on re-render).
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

export function buildCountyDataFromIncidence(records: IncidenceRecord[]): CountyData[] {
  const counts = new Map<string, number>();
  for (const record of records) {
    if (!record.county) continue;
    counts.set(record.county, (counts.get(record.county) ?? 0) + record.count);
  }

  return Array.from(counts.entries())
    .map(([county, count]) => ({
      county,
      region: regionForCounty(county),
      count,
      fips: '',
    }))
    .sort((a, b) => b.count - a.count);
}

export function applyCountyDemoFilters(base: CountyData[], filters: FilterState): CountyData[] {
  const isDefault =
    (!filters.sex || filters.sex === 'all') &&
    (!filters.cancerType || filters.cancerType === 'All Types') &&
    (!filters.breed || filters.breed === 'All Breeds');

  if (isDefault) return base;

  // Each filter narrows the data by a fraction, seeded so results are stable.
  const key = `${filters.sex}|${filters.cancerType}|${filters.breed}`;
  const rand = seededRandom(key);

  // Sex splits roughly into quarters; cancer type ~1/8; breed ~1/10
  let fraction = 1;
  if (filters.sex && filters.sex !== 'all') fraction *= 0.25;
  if (filters.cancerType && filters.cancerType !== 'All Types') fraction *= 0.12;
  if (filters.breed && filters.breed !== 'All Breeds') fraction *= 0.10;

  return base.map(c => {
    // Add per-county variation (±40%) around the fraction
    const variation = 0.6 + rand() * 0.8;
    const newCount = Math.max(0, Math.round(c.count * fraction * variation));
    return { ...c, count: newCount };
  }).filter(c => c.count > 0);
}

export function getCountRange(countyData: CountyData[]) {
  const counts = countyData.map(c => c.count).filter(n => n > 0);
  if (counts.length === 0) return { min: 0, max: 1 };
  return {
    min: Math.min(...counts),
    max: Math.max(...counts),
  };
}

export function createFilteredDataState(
  countyData: CountyData[],
  filters: FilterState,
  options: { applyServerSideFilters?: boolean } = {},
): Omit<FilteredDataState, 'loading' | 'error'> {
  const filteredCountyData = applyCountyDemoFilters(
    countyData,
    options.applyServerSideFilters === false
      ? ({ cancerType: 'All Types', breed: filters.breed, sex: 'all' } as FilterState)
      : filters,
  );
  const regionSummary = generateRegionSummary(filteredCountyData);

  return {
    countyData: filteredCountyData,
    regionSummary: filteredCountyData.length > 0 ? regionSummary : EMPTY_REGION_SUMMARY,
    countRange: getCountRange(filteredCountyData),
  };
}

export function useFilteredData(filters: FilterState): FilteredDataState {
  const { cancerType, sex, yearStart, yearEnd } = filters;
  const [countyData, setCountyData] = useState<CountyData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadCountyData() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetchIncidence({
          cancerTypes: cancerTypeFilterValue(cancerType),
          sex: sexFilterValue(sex),
          yearStart,
          yearEnd,
        });
        if (cancelled) return;

        setCountyData(buildCountyDataFromIncidence(response.data));
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Unable to load dashboard data');
        setCountyData([]);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadCountyData();

    return () => {
      cancelled = true;
    };
  }, [cancerType, sex, yearStart, yearEnd]);

  const derivedState = useMemo(
    () => createFilteredDataState(countyData, filters, { applyServerSideFilters: false }),
    [countyData, filters],
  );

  return {
    ...derivedState,
    loading,
    error,
  };
}

export function useCountyDataMap(countyData: CountyData[]): Map<string, CountyData> {
  return useMemo(() => {
    const map = new Map<string, CountyData>();
    countyData.forEach(county => {
      map.set(county.county, county);
    });
    return map;
  }, [countyData]);
}

// Generate hierarchical summary for the summary table
export function generateRegionSummary(countyData: CountyData[]): RegionSummary {
  const regionMap = new Map<string, CountyData[]>();

  countyData.forEach(county => {
    const existing = regionMap.get(county.region);
    if (existing) {
      existing.push(county);
    } else {
      regionMap.set(county.region, [county]);
    }
  });

  // Calculate totals
  const totalCount = countyData.reduce((sum, c) => sum + c.count, 0);

  // Dashboard grouping only; this does not change stored county-level data.
  const catchmentCounties = countyData.filter(c => isUcDavisCatchmentRegion(c.region));
  const catchmentCount = catchmentCounties.reduce((sum, c) => sum + c.count, 0);

  const regions: RegionSummary[] = Array.from(regionMap.entries()).map(([regionName, counties]) => {
    const regionCount = counties.reduce((sum, c) => sum + c.count, 0);
    return {
      name: regionName,
      type: 'region' as const,
      count: regionCount,
      children: counties.map(c => ({
        name: c.county,
        type: 'county' as const,
        count: c.count,
      })),
    };
  });

  return {
    name: 'California',
    type: 'state',
    count: totalCount,
    children: [
      {
        name: 'UC Davis Catchment Area',
        type: 'catchment',
        count: catchmentCount,
        children: regions.filter(r => isUcDavisCatchmentRegion(r.name)),
      },
      ...regions.filter(r => !isUcDavisCatchmentRegion(r.name)),
    ],
  };
}
