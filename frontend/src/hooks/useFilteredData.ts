import { useMemo, useEffect, useState } from 'react';
import type { FilterState, CountyData, RegionSummary } from '../types';
import { fetchIncidence, fetchCountiesGeoJSON } from '../api/client';

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
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Map frontend filter format to API format
        const apiFilters: Record<string, string[] | string | number | undefined> = {};

        if (filters.cancerType && filters.cancerType !== 'All Types') {
          apiFilters.cancerTypes = [filters.cancerType];
        }
        if (filters.breed && filters.breed !== 'All Breeds') {
          // Note: Backend currently doesn't filter by breed in geo endpoint, but we pass it anyway
        }
        if (filters.sex && filters.sex !== 'all') {
          apiFilters.sex = filters.sex;
        }

        // Fetch county GeoJSON which includes case counts and populations
        const geoData = await fetchCountiesGeoJSON(apiFilters as any);

        const counties: CountyData[] = geoData.features.map(feature => ({
          county: feature.properties.name,
          region: COUNTY_REGIONS[feature.properties.name] || 'Other',
          count: feature.properties.total_cases,
          population: feature.properties.population || 0,
          rate: feature.properties.cases_per_capita
            ? feature.properties.cases_per_capita / 10 // Convert from per 100k to per 10k
            : (feature.properties.population && feature.properties.population > 0
              ? Math.round((feature.properties.total_cases / feature.properties.population) * 10000 * 10) / 10
              : 0),
          fips: feature.properties.fips_code,
        }));

        setCountyData(counties);
      } catch (err) {
        console.error('Failed to load data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load data');
        setCountyData([]);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filters.cancerType, filters.breed, filters.sex]);

  const regionSummary = useMemo(() => {
    return generateRegionSummary(countyData);
  }, [countyData]);

  const rateRange = useMemo(() => {
    const rates = countyData.map(c => c.rate).filter(r => r > 0);
    if (rates.length === 0) return { min: 0, max: 100 };
    return {
      min: Math.min(...rates),
      max: Math.max(...rates),
    };
  }, [countyData]);

  return {
    countyData,
    regionSummary,
    rateRange,
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
  const totalPop = countyData.reduce((sum, c) => sum + c.population, 0);

  // Catchment area (Northern CA + Bay Area + Central Valley for UC Davis)
  const catchmentRegions = ['Bay Area', 'Northern CA', 'Central Valley'];
  const catchmentCounties = countyData.filter(c => catchmentRegions.includes(c.region));
  const catchmentCount = catchmentCounties.reduce((sum, c) => sum + c.count, 0);
  const catchmentPop = catchmentCounties.reduce((sum, c) => sum + c.population, 0);

  const regions: RegionSummary[] = Array.from(regionMap.entries()).map(([regionName, counties]) => {
    const regionCount = counties.reduce((sum, c) => sum + c.count, 0);
    const regionPop = counties.reduce((sum, c) => sum + c.population, 0);

    return {
      name: regionName,
      type: 'region' as const,
      count: regionCount,
      population: regionPop,
      rate: regionPop > 0 ? Math.round((regionCount / regionPop) * 10000 * 10) / 10 : 0,
      children: counties.map(c => ({
        name: c.county,
        type: 'county' as const,
        count: c.count,
        population: c.population,
        rate: c.rate,
      })),
    };
  });

  return {
    name: 'California',
    type: 'state',
    count: totalCount,
    population: totalPop,
    rate: totalPop > 0 ? Math.round((totalCount / totalPop) * 10000 * 10) / 10 : 0,
    children: [
      {
        name: 'UC Davis Catchment Area',
        type: 'catchment',
        count: catchmentCount,
        population: catchmentPop,
        rate: catchmentPop > 0 ? Math.round((catchmentCount / catchmentPop) * 10000 * 10) / 10 : 0,
        children: regions.filter(r => catchmentRegions.includes(r.name)),
      },
      ...regions.filter(r => !catchmentRegions.includes(r.name)),
    ],
  };
}
