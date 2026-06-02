import { describe, expect, it, type MockedFunction } from 'vitest';
import {
  cancelJob,
  fetchIncidence,
  fetchJobPreview,
  fetchJobs,
  fetchJson,
  fetchMe,
  filtersToParams,
  reviewJob,
  uploadCSV,
} from './client';

function fetchMock() {
  return globalThis.fetch as MockedFunction<typeof fetch>;
}

function jsonResponse(body: unknown, init: ResponseInit = {}) {
  return new Response(JSON.stringify(body), {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
}

describe('api client', () => {
  it('serializes filters into API query params', () => {
    const params = filtersToParams({
      species: ['Dog', 'Cat'],
      cancerTypes: ['Lymphoma', 'Melanoma'],
      counties: ['Sacramento', 'Yolo'],
      sex: 'female_spayed',
      yearStart: 2020,
      yearEnd: 2024,
    });

    expect(params.getAll('species')).toEqual(['Dog', 'Cat']);
    expect(params.getAll('cancer_type')).toEqual(['Lymphoma', 'Melanoma']);
    expect(params.getAll('county')).toEqual(['Sacramento', 'Yolo']);
    expect(params.get('sex')).toBe('female_spayed');
    expect(params.get('year_start')).toBe('2020');
    expect(params.get('year_end')).toBe('2024');
  });

  it('omits the query string when filters are empty or sex is all', async () => {
    fetchMock().mockImplementation(() => Promise.resolve(jsonResponse({ data: [], total: 0, filters_applied: {} })));

    await fetchIncidence({});
    await fetchIncidence({ sex: 'all' });

    expect(fetchMock()).toHaveBeenNthCalledWith(1, '/api/v1/incidence');
    expect(fetchMock()).toHaveBeenNthCalledWith(2, '/api/v1/incidence');
  });

  it('throws public API errors with the status code', async () => {
    fetchMock().mockResolvedValue(new Response('nope', { status: 503 }));

    await expect(fetchJson('/api/v1/dashboard/summary')).rejects.toThrow('API error: 503');
  });

  it('adds Authorization headers for auth requests', async () => {
    fetchMock().mockResolvedValue(jsonResponse({ email: 'admin@example.com', is_admin: true }));

    await fetchMe('access-token');

    expect(fetchMock()).toHaveBeenCalledWith('/api/v1/auth/me', {
      headers: { Authorization: 'Bearer access-token' },
    });
  });

  it('prefers backend auth error details and falls back for non-json responses', async () => {
    fetchMock()
      .mockResolvedValueOnce(jsonResponse({ detail: 'Not allowed' }, { status: 403 }))
      .mockResolvedValueOnce(new Response('plain text', { status: 401 }));

    await expect(fetchMe('token')).rejects.toThrow('Not allowed');
    await expect(fetchMe('token')).rejects.toThrow('HTTP 401');
  });

  it('uploads CSV files as FormData with auth and clinic name', async () => {
    fetchMock().mockResolvedValue(jsonResponse({ id: 12, status: 'pending_review' }));
    const dataset = new File(['a,b\n1,2'], 'dataset.csv', { type: 'text/csv' });

    await uploadCSV(dataset, 'upload-token', 'VMTH');

    const [, init] = fetchMock().mock.calls[0];
    expect(fetchMock()).toHaveBeenCalledWith('/api/v1/ingest/upload', expect.objectContaining({
      method: 'POST',
      headers: { Authorization: 'Bearer upload-token' },
    }));
    expect(init?.body).toBeInstanceOf(FormData);
    const formData = init?.body as FormData;
    expect(formData.get('dataset_a')).toBe(dataset);
    expect(formData.get('clinic_name')).toBe('VMTH');
  });

  it('serializes multiple job status filters', async () => {
    fetchMock().mockResolvedValue(jsonResponse([]));

    await fetchJobs('token', ['pending_review', 'processing']);

    expect(fetchMock()).toHaveBeenCalledWith('/api/v1/ingest/jobs?status=pending_review&status=processing', {
      headers: { Authorization: 'Bearer token' },
    });
  });

  it('posts review actions with a nullable rejection reason', async () => {
    fetchMock().mockResolvedValue(jsonResponse({ id: 8, status: 'processing' }));

    await reviewJob('token', 8, 'approve');

    expect(fetchMock()).toHaveBeenCalledWith('/api/v1/ingest/jobs/8/review', {
      method: 'POST',
      headers: {
        Authorization: 'Bearer token',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ action: 'approve', rejection_reason: null, model_folder: null, clinic_name: null }),
    });
  });

  it('returns job previews as text', async () => {
    fetchMock().mockResolvedValue(new Response('a,b\n1,2'));

    await expect(fetchJobPreview('token', 5, 'a')).resolves.toBe('a,b\n1,2');
  });

  it('posts to the job cancel endpoint', async () => {
    fetchMock().mockResolvedValue(jsonResponse({ id: 7, status: 'cancelled' }));

    await cancelJob('token', 7);

    expect(fetchMock()).toHaveBeenCalledWith('/api/v1/ingest/jobs/7/cancel', {
      method: 'POST',
      headers: { Authorization: 'Bearer token' },
    });
  });
});
