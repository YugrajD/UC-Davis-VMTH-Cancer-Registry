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
  total_cases: number;
  top_cancer?: string;
}

export interface GeoJSONResponse {
  type: "FeatureCollection";
  features: {
    type: "Feature";
    geometry: Record<string, unknown>;
    properties: GeoJSONFeatureProperties;
  }[];
}

export interface FilterOptions {
  species: { id: number; name: string }[];
  cancer_types: { id: number; name: string; description?: string }[];
  counties: { id: number; name: string; fips_code: string }[];
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

async function fetchJsonAuth<T>(url: string, token: string): Promise<T> {
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `API error: ${response.status}`);
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

// --- Breed Detail ---

export interface BreedDetail {
  breed: string;
  total_cases: number;
  sex_breakdown: { sex: string; count: number }[];
  cancer_types: { cancer_type: string; count: number }[];
  county_cases: { county_name: string; fips_code: string; count: number }[];
}

export async function fetchBreedDetail(breed: string): Promise<BreedDetail> {
  const params = new URLSearchParams({ breed });
  return fetchJson(`/api/v1/incidence/breed-detail?${params}`);
}

// --- CalEnviroScreen ---

import type { CalEnviroScreenData } from '../types';

export async function fetchCalEnviroScreen(): Promise<CalEnviroScreenData[]> {
  return fetchJson('/api/v1/geo/calenviroscreen');
}

// --- Auth ---

export interface MeResponse {
  email: string;
  is_admin: boolean;
}

export async function fetchMe(token: string): Promise<MeResponse> {
  return fetchJsonAuth('/api/v1/auth/me', token);
}

// --- Ingestion Jobs ---

export interface IngestionJob {
  id: number;
  uploaded_by_email: string;
  dataset_a_filename: string;
  dataset_b_filename: string;
  status: string;
  processing_stage?: string | null;
  reviewed_by_email?: string | null;
  reviewed_at?: string | null;
  rejection_reason?: string | null;
  ingestion_log_id?: number | null;
  processing_error?: string | null;
  result_summary?: {
    patients: number;
    diagnoses: number;
    avg_confidence: number | null;
    high_confidence: number;
    medium_confidence: number;
    low_confidence: number;
    top_cancer_types: { name: string; count: number }[];
  } | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface IngestionRowResult {
  row_number: number;
  anon_id: string;
  status: 'inserted' | 'skipped' | 'error';
  message?: string;
  cancer_type?: string;
  confidence?: number;
}

export interface IngestionResponse {
  total_rows: number;
  inserted: number;
  skipped: number;
  errors: number;
  warnings: string[];
  row_results: IngestionRowResult[];
}

export async function uploadCSV(
  datasetA: File,
  datasetB?: File,
  token?: string | null,
): Promise<IngestionJob> {
  const formData = new FormData();
  formData.append('dataset_a', datasetA);
  if (datasetB) formData.append('dataset_b', datasetB);

  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const response = await fetch('/api/v1/ingest/upload', {
    method: 'POST',
    headers,
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Upload failed: ${response.status}`);
  }

  return response.json();
}

export async function fetchJobs(token: string, statuses?: string[]): Promise<IngestionJob[]> {
  if (statuses && statuses.length > 0) {
    const params = new URLSearchParams();
    statuses.forEach(s => params.append('status', s));
    return fetchJsonAuth(`/api/v1/ingest/jobs?${params}`, token);
  }
  return fetchJsonAuth('/api/v1/ingest/jobs', token);
}

export async function fetchMyJobs(token: string): Promise<IngestionJob[]> {
  return fetchJsonAuth('/api/v1/ingest/jobs?mine=true', token);
}

export async function fetchJob(token: string, jobId: number): Promise<IngestionJob> {
  return fetchJsonAuth(`/api/v1/ingest/jobs/${jobId}`, token);
}

export async function fetchJobPreview(token: string, jobId: number, dataset: 'a' | 'b'): Promise<string> {
  const response = await fetch(`/api/v1/ingest/jobs/${jobId}/preview/${dataset}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Preview failed: ${response.status}`);
  }
  return response.text();
}

export async function cancelJob(token: string, jobId: number): Promise<IngestionJob> {
  const response = await fetch(`/api/v1/ingest/jobs/${jobId}/cancel`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Cancel failed: ${response.status}`);
  }
  return response.json();
}

export async function reviewJob(
  token: string,
  jobId: number,
  action: 'approve' | 'reject',
  rejectionReason?: string,
): Promise<IngestionJob> {
  const response = await fetch(`/api/v1/ingest/jobs/${jobId}/review`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      action,
      rejection_reason: rejectionReason || null,
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Review failed: ${response.status}`);
  }

  return response.json();
}
