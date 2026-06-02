import { useEffect, useMemo, useState } from 'react';
import type { FilterState, CountyData, RegionSummary, ZipCodeData } from '../types';
import {
  fetchIncidenceByZip,
  fetchPCCPByCounty,
  type IncidenceRecord,
  type PCCPResponse,
} from '../api/client';
import { isUcDavisCatchmentRegion, regionForCounty } from '../data/californiaRegions';

export interface FilteredDataState {
  countyData: CountyData[];
  regionSummary: RegionSummary;
  countRange: { min: number; max: number };
  loading: boolean;
  error: string | null;
  /** Always 0 in PCCP mode (kept for backward compat). */
  excludedCases: number;
  /** Total patients in the denominator across all counties. Zero while loading. */
  totalCases: number;
  /** Overall PCCP across all counties (cancer / total * 100). */
  overallPccp: number;
  /** Overall cancer patients count across all counties. */
  overallCancerPatients: number;
  /** Overall total patients (denominator) across all counties. */
  overallTotalPatients: number;
}

export interface FilteredZipCodeDataState {
  zipCodeData: ZipCodeData[];
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

export function buildCountyDataFromPCCP(response: PCCPResponse): {
  countyData: CountyData[];
  overallCancerPatients: number;
  overallTotalPatients: number;
  overallPccp: number;
} {
  const countyData: CountyData[] = response.data.map(r => ({
    county: r.county,
    region: regionForCounty(r.county),
    count: r.pccp,
    fips: '',
    totalPatients: r.total_patients,
  }));
  return {
    countyData,
    overallCancerPatients: response.overall_cancer_patients,
    overallTotalPatients: response.overall_total_patients,
    overallPccp: response.overall_pccp,
  };
}

// Deterministic pseudo-random from a string seed — stable across re-renders.
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

export function buildZipCodeDataFromIncidence(records: IncidenceRecord[]): ZipCodeData[] {
  const counts = new Map<string, number>();
  for (const record of records) {
    const zipCode = record.zip_code?.trim().slice(0, 5);
    if (!zipCode) continue;
    counts.set(zipCode, (counts.get(zipCode) ?? 0) + record.count);
  }

  return Array.from(counts.entries())
    .map(([zipCode, count]) => ({ zipCode, count }))
    .sort((a, b) => b.count - a.count);
}

function sexFilterValue(sex: FilterState['sex']) {
  return sex && sex !== 'all' ? sex : undefined;
}

function ageGroupFilterValue(ageGroup: FilterState['ageGroup']) {
  return ageGroup && ageGroup !== 'all' ? ageGroup : undefined;
}

function cancerTypeFilterValue(cancerType: string) {
  return cancerType && cancerType !== 'All Types' ? [cancerType] : undefined;
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

export function applyZipCodeDemoFilters(base: ZipCodeData[], filters: FilterState): ZipCodeData[] {
  const isDefault =
    (!filters.sex || filters.sex === 'all') &&
    (!filters.cancerType || filters.cancerType === 'All Types') &&
    (!filters.breed || filters.breed === 'All Breeds');

  if (isDefault) return base;

  const key = `zip|${filters.sex}|${filters.cancerType}|${filters.breed}`;
  const rand = seededRandom(key);

  let fraction = 1;
  if (filters.sex && filters.sex !== 'all') fraction *= 0.25;
  if (filters.cancerType && filters.cancerType !== 'All Types') fraction *= 0.12;
  if (filters.breed && filters.breed !== 'All Breeds') fraction *= 0.10;

  return base.map(z => {
    const variation = 0.6 + rand() * 0.8;
    const newCount = Math.max(0, Math.round(z.count * fraction * variation));
    return { ...z, count: newCount };
  }).filter(z => z.count > 0);
}

export function getCountRange(countyData: CountyData[]) {
  const counts = countyData.map(c => c.count).filter(n => n > 0);
  if (counts.length === 0) return { min: 0, max: 1 };
  return {
    min: Math.min(...counts),
    max: Math.max(...counts),
  };
}

export function getZipCodeCountRange(zipCodeData: ZipCodeData[]) {
  const counts = zipCodeData.map(z => z.count).filter(n => n > 0);
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
): Omit<FilteredDataState, 'loading' | 'error' | 'excludedCases' | 'totalCases' | 'overallPccp' | 'overallCancerPatients' | 'overallTotalPatients'> {
  const filteredCountyData = applyCountyDemoFilters(
    countyData,
    options.applyServerSideFilters === false
      ? ({ cancerType: 'All Types', breed: filters.breed, sex: 'all', ageGroup: 'all', rateType: filters.rateType } as FilterState)
      : filters,
  );
  const regionSummary = generateRegionSummary(filteredCountyData);

  return {
    countyData: filteredCountyData,
    regionSummary: filteredCountyData.length > 0 ? regionSummary : EMPTY_REGION_SUMMARY,
    countRange: getCountRange(filteredCountyData),
  };
}

export function createFilteredZipCodeDataState(
  zipCodeData: ZipCodeData[],
  filters: FilterState,
  options: { applyServerSideFilters?: boolean } = {},
): Omit<FilteredZipCodeDataState, 'loading' | 'error'> {
  const filteredZipCodeData = applyZipCodeDemoFilters(
    zipCodeData,
    options.applyServerSideFilters === false
      ? ({ cancerType: 'All Types', breed: filters.breed, sex: 'all', ageGroup: 'all', rateType: filters.rateType } as FilterState)
      : filters,
  );

  return {
    zipCodeData: filteredZipCodeData,
    countRange: getZipCodeCountRange(filteredZipCodeData),
  };
}

export function useFilteredData(filters: FilterState): FilteredDataState {
  const { sex, ageGroup, yearStart, yearEnd, cancerType } = filters;
  const [countyData, setCountyData] = useState<CountyData[]>([]);
  const [overallPccp, setOverallPccp] = useState(0);
  const [overallCancerPatients, setOverallCancerPatients] = useState(0);
  const [overallTotalPatients, setOverallTotalPatients] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetchPCCPByCounty({
          sex: sex && sex !== 'all' ? sex : undefined,
          ageGroup: ageGroup && ageGroup !== 'all' ? ageGroup : undefined,
          yearStart,
          yearEnd,
          cancerType: cancerType && cancerType !== 'All Types' ? cancerType : undefined,
        });
        if (cancelled) return;

        const { countyData: cd, overallCancerPatients: oc, overallTotalPatients: ot, overallPccp: op } =
          buildCountyDataFromPCCP(response);
        setCountyData(cd);
        setOverallCancerPatients(oc);
        setOverallTotalPatients(ot);
        setOverallPccp(op);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Unable to load dashboard data');
        setCountyData([]);
        setOverallCancerPatients(0);
        setOverallTotalPatients(0);
        setOverallPccp(0);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadData();

    return () => { cancelled = true; };
  }, [sex, ageGroup, yearStart, yearEnd, cancerType]);

  const regionSummary = useMemo(
    () => countyData.length > 0 ? generateRegionSummary(countyData) : EMPTY_REGION_SUMMARY,
    [countyData],
  );

  const countRange = useMemo(() => getCountRange(countyData), [countyData]);

  return {
    countyData,
    regionSummary,
    countRange,
    loading,
    error,
    excludedCases: 0,
    totalCases: overallTotalPatients,
    overallPccp,
    overallCancerPatients,
    overallTotalPatients,
  };
}

export function useZipCodeData(filters: FilterState): FilteredZipCodeDataState {
  const { cancerType, sex, ageGroup, yearStart, yearEnd } = filters;
  const [zipCodeData, setZipCodeData] = useState<ZipCodeData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadZipCodeData() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetchIncidenceByZip({
          cancerTypes: cancerTypeFilterValue(cancerType),
          sex: sexFilterValue(sex),
          ageGroup: ageGroupFilterValue(ageGroup),
          yearStart,
          yearEnd,
        });
        if (cancelled) return;

        setZipCodeData(buildZipCodeDataFromIncidence(response.data));
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Unable to load ZIP code data');
        setZipCodeData([]);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadZipCodeData();

    return () => {
      cancelled = true;
    };
  }, [cancerType, sex, ageGroup, yearStart, yearEnd]);

  const derivedState = useMemo(
    () => createFilteredZipCodeDataState(zipCodeData, filters, { applyServerSideFilters: false }),
    [zipCodeData, filters],
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

// Aggregate county counts: simple sum for raw data, weighted average for PCCP data.
function aggregateCount(counties: CountyData[]): number {
  if (counties.length === 0) return 0;
  const totalPts = counties.reduce((sum, c) => sum + (c.totalPatients ?? 0), 0);
  if (totalPts === 0) return counties.reduce((sum, c) => sum + c.count, 0);
  return counties.reduce((sum, c) => sum + c.count * (c.totalPatients ?? 0), 0) / totalPts;
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

  const totalCount = aggregateCount(countyData);

  const catchmentCounties = countyData.filter(c => isUcDavisCatchmentRegion(c.region));
  const catchmentCount = aggregateCount(catchmentCounties);

  const regions: RegionSummary[] = Array.from(regionMap.entries()).map(([regionName, counties]) => {
    return {
      name: regionName,
      type: 'region' as const,
      count: aggregateCount(counties),
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
