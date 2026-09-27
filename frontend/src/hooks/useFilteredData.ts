import { useEffect, useMemo, useState } from 'react';
import type { FilterState, CountyData, RateType, RegionSummary, ZipCodeData } from '../types';
import {
  fetchPCCPByCounty,
  fetchPCCPByZip,
  type IncidenceRecord,
  type PCCPResponse,
  type PCCPZipResponse,
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
    casePatients: r.cancer_patients,
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

export function buildZipCodeDataFromPCCP(response: PCCPZipResponse): ZipCodeData[] {
  return response.data.map(r => ({
    zipCode: r.zip_code,
    count: r.pccp,
    casePatients: r.cancer_patients,
    totalPatients: r.total_patients,
  }));
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

/** Shape shared by CountyData and ZipCodeData — anything with a PCCP-mode count plus numerator/denominator. */
interface RateSource {
  count: number;
  casePatients?: number;
  totalPatients?: number;
}

/** The value a county/ZIP contributes for the currently selected map-color metric. */
export function valueForRate(source: RateSource | undefined, rateType: RateType): number {
  if (!source) return 0;
  switch (rateType) {
    case 'numerator':
      return source.casePatients ?? 0;
    case 'denominator':
      return source.totalPatients ?? 0;
    case 'pccp':
    default:
      return source.count;
  }
}

/** Min/max range for the currently selected map-color metric, for the color scale domain. */
export function getCountRangeForRate(data: RateSource[], rateType: RateType) {
  const values = data.map(c => valueForRate(c, rateType)).filter(n => n > 0);
  if (values.length === 0) return { min: 0, max: 1 };
  return {
    min: Math.min(...values),
    max: Math.max(...values),
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
  const { sex, ageGroup, yearStart, yearEnd, cancerType, breed } = filters;
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
          breed: breed && breed !== 'All Breeds' ? breed : undefined,
        });
        const pccpData = buildCountyDataFromPCCP(response);
        if (cancelled) return;

        setCountyData(pccpData.countyData);
        setOverallCancerPatients(pccpData.overallCancerPatients);
        setOverallTotalPatients(pccpData.overallTotalPatients);
        setOverallPccp(pccpData.overallPccp);
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
  }, [sex, ageGroup, yearStart, yearEnd, cancerType, breed]);

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
  const { cancerType, sex, ageGroup, yearStart, yearEnd, breed } = filters;
  const [zipCodeData, setZipCodeData] = useState<ZipCodeData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadZipCodeData() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetchPCCPByZip({
          sex: sex && sex !== 'all' ? sex : undefined,
          ageGroup: ageGroup && ageGroup !== 'all' ? ageGroup : undefined,
          yearStart,
          yearEnd,
          cancerType: cancerType && cancerType !== 'All Types' ? cancerType : undefined,
          breed: breed && breed !== 'All Breeds' ? breed : undefined,
        });
        const zd = buildZipCodeDataFromPCCP(response);
        if (cancelled) return;

        setZipCodeData(zd);
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
  }, [cancerType, sex, ageGroup, yearStart, yearEnd, breed]);

  const countRange = useMemo(() => getZipCodeCountRange(zipCodeData), [zipCodeData]);

  return {
    zipCodeData,
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

// Aggregate a set of counties into a region-level rollup. When every county
// carries numerator/denominator data (PCCP mode), the region's PCCP is the
// true sum(numerator)/sum(denominator) — not an average of county PCCPs —
// so the three displayed columns (numerator, denominator, PCCP) reconcile.
// Falls back to a simple sum of `count` for raw incidence data, which has
// no numerator/denominator.
function aggregateRegion(counties: CountyData[]): {
  count: number;
  casePatients?: number;
  totalPatients?: number;
} {
  if (counties.length === 0) return { count: 0 };
  const hasPccpData = counties.every(
    c => c.casePatients !== undefined && c.totalPatients !== undefined,
  );
  if (!hasPccpData) {
    return { count: counties.reduce((sum, c) => sum + c.count, 0) };
  }
  const casePatients = counties.reduce((sum, c) => sum + (c.casePatients ?? 0), 0);
  const totalPatients = counties.reduce((sum, c) => sum + (c.totalPatients ?? 0), 0);
  const count = totalPatients > 0 ? (casePatients / totalPatients) * 100 : 0;
  return { count, casePatients, totalPatients };
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

  const total = aggregateRegion(countyData);

  const catchmentCounties = countyData.filter(c => isUcDavisCatchmentRegion(c.region));
  const catchmentTotal = aggregateRegion(catchmentCounties);

  const regions: RegionSummary[] = Array.from(regionMap.entries()).map(([regionName, counties]) => {
    const regionTotal = aggregateRegion(counties);
    return {
      name: regionName,
      type: 'region' as const,
      count: regionTotal.count,
      casePatients: regionTotal.casePatients,
      totalPatients: regionTotal.totalPatients,
      children: counties.map(c => ({
        name: c.county,
        type: 'county' as const,
        count: c.count,
        casePatients: c.casePatients,
        totalPatients: c.totalPatients,
      })),
    };
  });

  return {
    name: 'California',
    type: 'state',
    count: total.count,
    casePatients: total.casePatients,
    totalPatients: total.totalPatients,
    children: [
      {
        name: 'UC Davis Catchment Area',
        type: 'catchment',
        count: catchmentTotal.count,
        casePatients: catchmentTotal.casePatients,
        totalPatients: catchmentTotal.totalPatients,
        children: regions.filter(r => isUcDavisCatchmentRegion(r.name)),
      },
      ...regions.filter(r => !isUcDavisCatchmentRegion(r.name)),
    ],
  };
}
