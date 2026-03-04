import { useMemo, useEffect, useState } from 'react';
import type { FilterState, CountyData, RegionSummary } from '../types';
import { MOCK_COUNTY_DATA } from '../data/mockData';

// Region mapping for UC Davis catchment area
const COUNTY_REGIONS: Record<string, string> = {
  // Bay Area
  'Alameda': 'Bay Area',
  'Contra Costa': 'Bay Area',
  'Marin': 'Bay Area',
  'San Francisco': 'Bay Area',
  'San Mateo': 'Bay Area',
  'Santa Clara': 'Bay Area',
  'Sonoma': 'Bay Area',
  'Napa': 'Bay Area',
  'Solano': 'Bay Area',
  // Northern CA
  'Butte': 'Northern CA',
  'Shasta': 'Northern CA',
  'Humboldt': 'Northern CA',
  'Mendocino': 'Northern CA',
  'Del Norte': 'Northern CA',
  'Nevada': 'Northern CA',
  'Yuba': 'Northern CA',
  'Sutter': 'Northern CA',
  'Glenn': 'Northern CA',
  'Colusa': 'Northern CA',
  // Central Valley
  'Sacramento': 'Central Valley',
  'San Joaquin': 'Central Valley',
  'Fresno': 'Central Valley',
  'Stanislaus': 'Central Valley',
  'Kern': 'Central Valley',
  'Yolo': 'Central Valley',
  'Placer': 'Central Valley',
  'El Dorado': 'Central Valley',
  'Amador': 'Central Valley',
  // Central Coast
  'Monterey': 'Central Coast',
  'Santa Cruz': 'Central Coast',
  'San Luis Obispo': 'Central Coast',
  'Santa Barbara': 'Central Coast',
  // Southern CA
  'Los Angeles': 'Southern CA',
  'Orange': 'Southern CA',
  'San Diego': 'Southern CA',
  'Riverside': 'Southern CA',
  'San Bernardino': 'Southern CA',
  'Ventura': 'Southern CA',
};

export function useFilteredData(filters: FilterState) {
  const [countyData, setCountyData] = useState<CountyData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // For the presentation, use the static mock county data
    // so the dashboard always matches the screenshot values.
    setLoading(true);
    setError(null);
    setCountyData(MOCK_COUNTY_DATA);
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
