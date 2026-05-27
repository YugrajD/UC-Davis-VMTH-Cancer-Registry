// API client for the VMTH Cancer Registry backend

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

const API_BASE_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

function apiUrl(path: string): string {
  if (!API_BASE_URL) return path;
  return `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`;
}

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
  const response = await fetch(apiUrl(url));
  if (!response.ok) {
    throw new ApiError(response.status, `API error: ${response.status}`);
  }
  return response.json();
}

async function fetchJsonAuth<T>(url: string, token: string): Promise<T> {
  const response = await fetch(apiUrl(url), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    // FastAPI returns `detail` as a string for HTTPException but as an array of
    // validation-error objects for 422 Unprocessable Entity.  Normalise to string.
    const raw = err.detail;
    const msg = typeof raw === 'string'
      ? raw
      : Array.isArray(raw)
        ? raw.map((e: { msg?: string }) => e.msg ?? JSON.stringify(e)).join('; ')
        : `API error: ${response.status}`;
    throw new ApiError(response.status, msg);
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

// --- Yearly trends ---

export interface TrendPointApi {
  year: number;
  count: number;
  deceased: number | null;
  alive: number | null;
}

export interface TrendSeriesApi {
  name: string;
  data: TrendPointApi[];
}

export interface TrendsResponse {
  series: TrendSeriesApi[];
}

export async function fetchYearlyTrends(filters: FilterParams = {}): Promise<TrendsResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/trends/yearly?${params}` : '/api/v1/trends/yearly';
  return fetchJson(url);
}

export async function fetchTrendsByCancerType(filters: FilterParams = {}): Promise<TrendsResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/trends/by-cancer-type?${params}` : '/api/v1/trends/by-cancer-type';
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
  is_uploader: boolean;
  is_reviewer: boolean;
}

export async function fetchMe(token: string): Promise<MeResponse> {
  return fetchJsonAuth('/api/v1/auth/me', token);
}

// --- User-role admin panel ---

export interface UserRoles {
  email: string;
  is_admin: boolean;
  is_uploader: boolean;
  is_reviewer: boolean;
  updated_by_email: string | null;
  updated_at: string | null;
  /** False when no DB row exists yet — values come from env-fallback or defaults. */
  persisted: boolean;
}

/** Treat empty input and bare strings without "@" as invalid client-side. */
export function normalizeEmail(raw: string): string {
  return raw.trim().toLowerCase();
}

export function isValidEmail(raw: string): boolean {
  const e = normalizeEmail(raw);
  return e.length > 0 && e.length <= 255 && e.includes('@');
}

export async function fetchUserRoles(token: string, email: string): Promise<UserRoles> {
  const normalized = normalizeEmail(email);
  return fetchJsonAuth(
    `/api/v1/admin/users/${encodeURIComponent(normalized)}/roles`,
    token,
  );
}

export async function updateUserRoles(
  token: string,
  email: string,
  roles: { is_admin: boolean; is_uploader: boolean; is_reviewer: boolean },
): Promise<UserRoles> {
  const normalized = normalizeEmail(email);
  const response = await fetch(
    apiUrl(`/api/v1/admin/users/${encodeURIComponent(normalized)}/roles`),
    {
      method: 'PUT',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(roles),
    },
  );
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Update failed: ${response.status}`);
  }
  return response.json();
}

// --- Ingestion Jobs ---

export interface IngestionJob {
  id: number;
  uploaded_by_email: string;
  dataset_a_filename: string;
  status: string;
  processing_stage?: string | null;
  model_folder?: string | null;
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
  dataset: File,
  token: string,
): Promise<IngestionJob> {
  const formData = new FormData();
  formData.append('dataset_a', dataset);

  const response = await fetch(apiUrl('/api/v1/ingest/upload'), {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
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

export async function fetchJobPreview(token: string, jobId: number): Promise<string> {
  const response = await fetch(apiUrl(`/api/v1/ingest/jobs/${jobId}/preview`), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Preview failed: ${response.status}`);
  }
  return response.text();
}

export async function cancelJob(token: string, jobId: number): Promise<IngestionJob> {
  const response = await fetch(apiUrl(`/api/v1/ingest/jobs/${jobId}/cancel`), {
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
  modelFolder?: string,
): Promise<IngestionJob> {
  const response = await fetch(apiUrl(`/api/v1/ingest/jobs/${jobId}/review`), {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      action,
      rejection_reason: rejectionReason || null,
      model_folder: modelFolder || null,
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Review failed: ${response.status}`);
  }

  return response.json();
}

export async function fetchAvailableModels(token: string): Promise<string[]> {
  const data = await fetchJsonAuth('/api/v1/ingest/models', token);
  return (data as { models: string[] }).models;
}


// --- Per-diagnosis review queue (admin-only) ---

export interface PendingDiagnosis {
  id: number;
  patient_anon_id: string | null;
  cancer_type_id: number;
  cancer_type_name: string;
  icd_o_code: string | null;
  predicted_term: string | null;
  confidence: number | null;
  top2_margin: number | null;
  prediction_method: string | null;
  diagnosis_index: number | null;
  review_status: 'pending' | 'confirmed' | 'corrected' | 'rejected';
  ingestion_job_id: number | null;
  job_filename: string | null;
  job_created_at: string | null;
}

export interface DiagnosisReviewEvent {
  id: number;
  actor_email: string;
  action: string;
  from_status: string | null;
  to_status: string;
  cancer_type_id_before: number | null;
  cancer_type_id_after: number | null;
  icd_o_code_before: string | null;
  icd_o_code_after: string | null;
  notes: string | null;
  created_at: string;
}

export interface DiagnosisDetail extends PendingDiagnosis {
  original_cancer_type_id: number | null;
  original_icd_o_code: string | null;
  original_predicted_term: string | null;
  /** Raw source text PetBERT classified — null for legacy rows. */
  original_text: string | null;
  reviewed_by_email: string | null;
  reviewed_at: string | null;
  reviewer_notes: string | null;
  events: DiagnosisReviewEvent[];
}

export type ReviewActionKind = 'confirm' | 'correct' | 'reject';

export interface ReviewActionPayload {
  action: ReviewActionKind;
  cancer_type_name?: string;
  icd_o_code?: string;
  predicted_term?: string;
  notes?: string;
}

export async function fetchPendingDiagnoses(
  token: string,
  params: {
    limit?: number;
    offset?: number;
    cancer_type_id?: number;
    method?: string;
    max_confidence?: number;
    ingestion_job_id?: number;
    year?: number;
    patient_id?: string;
    uploaded_by?: string;
  } = {},
): Promise<PendingDiagnosis[]> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) qs.append(k, String(v));
  }
  const url = `/api/v1/diagnoses/pending${qs.toString() ? `?${qs}` : ''}`;
  return fetchJsonAuth(url, token);
}

export async function fetchAllDiagnoses(
  token: string,
  params: {
    status?: string;
    limit?: number;
    offset?: number;
    ingestion_job_id?: number;
    year?: number;
    patient_id?: string;
    uploaded_by?: string;
  } = {},
): Promise<PendingDiagnosis[]> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) qs.append(k, String(v));
  }
  const url = `/api/v1/diagnoses${qs.toString() ? `?${qs}` : ''}`;
  return fetchJsonAuth(url, token);
}

export async function fetchPendingCount(token: string): Promise<{ count: number }> {
  return fetchJsonAuth('/api/v1/diagnoses/pending/count', token);
}

export async function fetchDiagnosisUploaders(token: string): Promise<string[]> {
  return fetchJsonAuth('/api/v1/diagnoses/uploaders', token);
}

export async function fetchDiagnosisDetail(
  token: string,
  diagnosisId: number,
): Promise<DiagnosisDetail> {
  return fetchJsonAuth(`/api/v1/diagnoses/${diagnosisId}`, token);
}

export async function reviewDiagnosis(
  token: string,
  diagnosisId: number,
  payload: ReviewActionPayload,
): Promise<DiagnosisDetail> {
  const response = await fetch(apiUrl(`/api/v1/diagnoses/${diagnosisId}/review`), {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Review failed: ${response.status}`);
  }

  return response.json();
}


// --- Role Requests ---

export interface RoleRequest {
  id: number;
  email: string;
  requested_role: string;
  status: string;
  reason: string | null;
  resolved_by_email: string | null;
  resolved_at: string | null;
  created_at: string;
}

export async function submitRoleRequest(
  token: string,
  requested_role: string,
  reason?: string,
): Promise<RoleRequest> {
  const response = await fetch(apiUrl('/api/v1/role-requests/'), {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ requested_role, reason: reason || null }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Request failed: ${response.status}`);
  }

  return response.json();
}

export async function fetchMyRoleRequests(token: string): Promise<RoleRequest[]> {
  return fetchJsonAuth('/api/v1/role-requests/mine', token);
}

export async function fetchPendingRoleRequests(token: string): Promise<RoleRequest[]> {
  return fetchJsonAuth('/api/v1/role-requests/pending', token);
}

export async function fetchPendingRoleRequestCount(token: string): Promise<{ count: number }> {
  return fetchJsonAuth('/api/v1/role-requests/pending/count', token);
}

export async function resolveRoleRequest(
  token: string,
  requestId: number,
  action: 'approve' | 'deny',
): Promise<RoleRequest> {
  const response = await fetch(apiUrl(`/api/v1/role-requests/${requestId}/resolve`), {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ action }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Resolve failed: ${response.status}`);
  }

  return response.json();
}

// --- Export Requests ---

export interface ExportRequest {
  id: number;
  email: string;
  status: string;
  reason: string | null;
  resolved_by_email: string | null;
  resolved_at: string | null;
  created_at: string;
}

export async function submitExportRequest(
  token: string,
  reason?: string,
): Promise<ExportRequest> {
  const response = await fetch(apiUrl('/api/v1/export-requests/'), {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ reason: reason || null }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Request failed: ${response.status}`);
  }

  return response.json();
}

export async function fetchMyExportRequests(token: string): Promise<ExportRequest[]> {
  return fetchJsonAuth('/api/v1/export-requests/mine', token);
}

export async function fetchPendingExportRequests(token: string): Promise<ExportRequest[]> {
  return fetchJsonAuth('/api/v1/export-requests/pending', token);
}

export async function fetchPendingExportRequestCount(token: string): Promise<{ count: number }> {
  return fetchJsonAuth('/api/v1/export-requests/pending/count', token);
}

export async function resolveExportRequest(
  token: string,
  requestId: number,
  action: 'approve' | 'deny',
): Promise<ExportRequest> {
  const response = await fetch(apiUrl(`/api/v1/export-requests/${requestId}/resolve`), {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ action }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Resolve failed: ${response.status}`);
  }

  return response.json();
}

export interface ExportFilters {
  cancerType?: string;
  county?: string;
  zipCode?: string;
  sex?: string;
  breed?: string;
  yearStart?: number;
  yearEnd?: number;
}

export async function downloadExportCsv(token: string, filters?: ExportFilters): Promise<Blob> {
  const params = new URLSearchParams();
  if (filters?.cancerType) params.append('cancer_type', filters.cancerType);
  if (filters?.county) params.append('county', filters.county);
  if (filters?.zipCode) params.append('zip_code', filters.zipCode);
  if (filters?.sex) params.append('sex', filters.sex);
  if (filters?.breed) params.append('breed', filters.breed);
  if (filters?.yearStart) params.append('year_start', String(filters.yearStart));
  if (filters?.yearEnd) params.append('year_end', String(filters.yearEnd));

  const url = params.toString()
    ? `/api/v1/export-requests/download?${params}`
    : '/api/v1/export-requests/download';
  const response = await fetch(apiUrl(url), {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(err.detail || `Download failed: ${response.status}`);
  }

  return response.blob();
}
