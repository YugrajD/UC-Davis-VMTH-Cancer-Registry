// API client for the VMTH Cancer Registry backend

export interface DashboardSummary {
  total_cases: number;
  total_patients: number;
  total_counties: number;
  year_range: number[];
  species_breakdown: { species: string; count: number; percentage: number }[];
  top_cancers: { cancer_type: string; count: number }[];
  top_county: string;
  top_county_cases: number;
}

export interface IncidenceRecord {
  cancer_type: string;
  county?: string;
  species?: string;
  breed?: string;
  year?: number;
  count: number;
}

export interface IncidenceResponse {
  data: IncidenceRecord[];
  total: number;
  filters_applied: Record<string, unknown>;
}

export interface GeoJSONFeatureProperties {
  name: string;
  fips_code: string;
  population?: number;
  total_cases: number;
  cases_per_capita?: number;
  top_cancer?: string;
}

export interface GeoJSONResponse {
  type: "FeatureCollection";
  features: {
    type: "Feature";
    geometry: GeoJSON.Geometry;
    properties: GeoJSONFeatureProperties;
  }[];
}

export interface FilterOptions {
  species: { id: number; name: string }[];
  cancer_types: { id: number; name: string; description?: string }[];
  counties: { id: number; name: string; fips_code: string; population?: number }[];
  breeds: { id: number; species_id: number; name: string }[];
  year_range: number[];
}

interface FilterParams {
  species?: string[];
  cancerTypes?: string[];
  counties?: string[];
  sex?: string;
  yearStart?: number;
  yearEnd?: number;
}

function filtersToParams(filters: FilterParams): URLSearchParams {
  const params = new URLSearchParams();
  if (filters.species?.length) {
    filters.species.forEach(s => params.append('species', s));
  }
  if (filters.cancerTypes?.length) {
    filters.cancerTypes.forEach(ct => params.append('cancer_type', ct));
  }
  if (filters.counties?.length) {
    filters.counties.forEach(c => params.append('county', c));
  }
  if (filters.sex && filters.sex !== 'all') {
    params.append('sex', filters.sex);
  }
  if (filters.yearStart) {
    params.append('year_start', String(filters.yearStart));
  }
  if (filters.yearEnd) {
    params.append('year_end', String(filters.yearEnd));
  }
  return params;
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}

export async function fetchDashboardSummary(): Promise<DashboardSummary> {
  return fetchJson('/api/v1/dashboard/summary');
}

export async function fetchFilterOptions(): Promise<FilterOptions> {
  return fetchJson('/api/v1/dashboard/filters');
}

export async function fetchIncidence(filters: FilterParams = {}): Promise<IncidenceResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/incidence?${params}` : '/api/v1/incidence';
  return fetchJson(url);
}

export async function fetchIncidenceByCancerType(filters: FilterParams = {}): Promise<IncidenceResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/incidence/by-cancer-type?${params}` : '/api/v1/incidence/by-cancer-type';
  return fetchJson(url);
}

export async function fetchIncidenceBySpecies(filters: FilterParams = {}): Promise<IncidenceResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/incidence/by-species?${params}` : '/api/v1/incidence/by-species';
  return fetchJson(url);
}

export async function fetchIncidenceByBreed(filters: FilterParams = {}): Promise<IncidenceResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/incidence/by-breed?${params}` : '/api/v1/incidence/by-breed';
  return fetchJson(url);
}

export async function fetchCountiesGeoJSON(filters: FilterParams = {}): Promise<GeoJSONResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/geo/counties?${params}` : '/api/v1/geo/counties';
  return fetchJson(url);
}
