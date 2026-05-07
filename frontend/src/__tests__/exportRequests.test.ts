import { describe, it, expect } from 'vitest';
import {
  submitExportRequest,
  fetchMyExportRequests,
  fetchPendingExportRequests,
  fetchPendingExportRequestCount,
  resolveExportRequest,
  downloadExportCsv,
} from '../api/client';

// ---------------------------------------------------------------------------
// Export request API functions — verify they are exported and callable
// ---------------------------------------------------------------------------

describe('Export request API functions', () => {
  it('submitExportRequest is a function', () => {
    expect(typeof submitExportRequest).toBe('function');
  });

  it('fetchMyExportRequests is a function', () => {
    expect(typeof fetchMyExportRequests).toBe('function');
  });

  it('fetchPendingExportRequests is a function', () => {
    expect(typeof fetchPendingExportRequests).toBe('function');
  });

  it('fetchPendingExportRequestCount is a function', () => {
    expect(typeof fetchPendingExportRequestCount).toBe('function');
  });

  it('resolveExportRequest is a function', () => {
    expect(typeof resolveExportRequest).toBe('function');
  });

  it('downloadExportCsv is a function', () => {
    expect(typeof downloadExportCsv).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Export request API functions — verify argument expectations
// ---------------------------------------------------------------------------

describe('Export request API function signatures', () => {
  it('submitExportRequest rejects without a token (network error)', async () => {
    // Calling with an empty token will attempt a fetch that fails in Node
    await expect(submitExportRequest('')).rejects.toThrow();
  });

  it('fetchMyExportRequests rejects without a valid endpoint', async () => {
    await expect(fetchMyExportRequests('')).rejects.toThrow();
  });

  it('fetchPendingExportRequests rejects without a valid endpoint', async () => {
    await expect(fetchPendingExportRequests('')).rejects.toThrow();
  });

  it('fetchPendingExportRequestCount rejects without a valid endpoint', async () => {
    await expect(fetchPendingExportRequestCount('')).rejects.toThrow();
  });

  it('resolveExportRequest rejects without a valid endpoint', async () => {
    await expect(resolveExportRequest('', 1, 'approve')).rejects.toThrow();
  });

  it('downloadExportCsv rejects without a valid endpoint', async () => {
    await expect(downloadExportCsv('')).rejects.toThrow();
  });
});
