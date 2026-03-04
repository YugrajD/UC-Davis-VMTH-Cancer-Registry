import { useMemo, useEffect, useState } from 'react';
import type { FilterState, CountyData, RegionSummary } from '../types';
import { MOCK_COUNTY_DATA } from '../data/mockData';

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

function applyFilters(base: CountyData[], filters: FilterState): CountyData[] {
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

export function useFilteredData(filters: FilterState) {
  const [countyData, setCountyData] = useState<CountyData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setCountyData(applyFilters(MOCK_COUNTY_DATA, filters));
    setLoading(false);
  }, [filters.cancerType, filters.breed, filters.sex]);

  const regionSummary = useMemo(() => {
    return generateRegionSummary(countyData);
  }, [countyData]);

  const countRange = useMemo(() => {
    const counts = countyData.map(c => c.count).filter(n => n > 0);
    if (counts.length === 0) return { min: 0, max: 1 };
    return {
      min: Math.min(...counts),
      max: Math.max(...counts),
    };
  }, [countyData]);

  return {
    countyData,
    regionSummary,
    countRange,
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
function generateRegionSummary(countyData: CountyData[]): RegionSummary {
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

  // Catchment area (Northern CA + Bay Area + Central Valley for UC Davis)
  const catchmentRegions = ['Bay Area', 'Northern CA', 'Central Valley'];
  const catchmentCounties = countyData.filter(c => catchmentRegions.includes(c.region));
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
        children: regions.filter(r => catchmentRegions.includes(r.name)),
      },
      ...regions.filter(r => !catchmentRegions.includes(r.name)),
    ],
  };
}
