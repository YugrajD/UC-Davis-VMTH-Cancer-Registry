import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { fetchJobs, reviewJob, fetchJobPreview, type IngestionJob } from '../../api/client';

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending_review: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    processing: 'bg-blue-100 text-blue-800 border-blue-200',
    completed: 'bg-green-100 text-green-800 border-green-200',
    failed: 'bg-red-100 text-red-800 border-red-200',
    rejected: 'bg-red-100 text-red-800 border-red-200',
  };
  const labels: Record<string, string> = {
    pending_review: 'Pending Review',
    processing: 'Processing',
    completed: 'Completed',
    failed: 'Failed',
    rejected: 'Rejected',
  };

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${colors[status] || 'bg-gray-100 text-gray-800 border-gray-200'}`}>
      {labels[status] || status}
    </span>
  );
}

function PreviewModal({ content, filename, onClose }: { content: string; filename: string; onClose: () => void }) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const lines = content.split(/\r?\n/).filter(l => l.trim());
  const headers = lines[0]?.split(',') || [];
  const rows = lines.slice(1).map(l => l.split(','));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/50" />
      <div className="relative bg-white rounded-xl shadow-2xl flex flex-col w-full max-w-5xl max-h-[90vh]" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 shrink-0">
          <h2 className="text-base font-semibold text-[var(--color-text-primary)]">{filename}</h2>
          <button onClick={onClose} className="p-1.5 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="overflow-auto flex-1 min-h-0">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 sticky top-0 z-10">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-400 w-12">#</th>
                {headers.map((h, i) => (
                  <th key={i} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">{h.trim()}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {rows.slice(0, 200).map((row, ri) => (
                <tr key={ri} className="hover:bg-gray-50">
                  <td className="px-4 py-1.5 text-xs text-gray-400">{ri + 1}</td>
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-4 py-1.5 text-gray-700 whitespace-nowrap max-w-[300px] truncate" title={cell.trim()}>
                      {cell.trim() || <span className="text-gray-300">&mdash;</span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export function AdminQueue() {
  const { getAccessToken } = useAuth();
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [rejectJobId, setRejectJobId] = useState<number | null>(null);
  const [rejectReason, setRejectReason] = useState('');
  const [preview, setPreview] = useState<{ content: string; filename: string } | null>(null);
  const [previewLoading, setPreviewLoading] = useState<string | null>(null); // "jobId-dataset" key

  const loadJobs = useCallback(async () => {
    try {
      const token = await getAccessToken();
      if (!token) return;
      const data = await fetchJobs(token);
      setJobs(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load jobs');
    } finally {
      setLoading(false);
    }
  }, [getAccessToken]);

  useEffect(() => {
    loadJobs();
  }, [loadJobs]);

  // Auto-poll when any job is processing
  useEffect(() => {
    const hasProcessing = jobs.some(j => j.status === 'processing');
    if (!hasProcessing) return;
    const interval = setInterval(loadJobs, 10000);
    return () => clearInterval(interval);
  }, [jobs, loadJobs]);

  const handleApprove = async (jobId: number) => {
    if (!confirm('Approve this upload for processing?')) return;
    setActionLoading(jobId);
    try {
      const token = await getAccessToken();
      if (!token) return;
      await reviewJob(token, jobId, 'approve');
      await loadJobs();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Approve failed');
    } finally {
      setActionLoading(null);
    }
  };

  const handleReject = async (jobId: number) => {
    setActionLoading(jobId);
    try {
      const token = await getAccessToken();
      if (!token) return;
      await reviewJob(token, jobId, 'reject', rejectReason || undefined);
      setRejectJobId(null);
      setRejectReason('');
      await loadJobs();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Reject failed');
    } finally {
      setActionLoading(null);
    }
  };

  const handlePreview = async (jobId: number, dataset: 'a' | 'b', filename: string) => {
    const key = `${jobId}-${dataset}`;
    setPreviewLoading(key);
    try {
      const token = await getAccessToken();
      if (!token) return;
      const content = await fetchJobPreview(token, jobId, dataset);
      setPreview({ content, filename });
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Preview failed');
    } finally {
      setPreviewLoading(null);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-12 flex flex-col items-center justify-center">
        <div className="w-8 h-8 border-4 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
        <p className="mt-4 text-sm text-[var(--color-text-secondary)]">Loading review queue...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-sm text-red-800">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Review Queue</h2>
          <button
            onClick={loadJobs}
            className="text-sm text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium"
          >
            Refresh
          </button>
        </div>
      </div>

      {jobs.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <p className="text-sm text-[var(--color-text-secondary)]">No uploads in the queue.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {jobs.map((job) => (
            <div key={job.id} className="bg-white rounded-lg border border-gray-200 p-5">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-sm font-semibold text-[var(--color-text-primary)]">
                      Job #{job.id}
                    </span>
                    <StatusBadge status={job.status} />
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs text-[var(--color-text-secondary)]">
                    <p>Uploaded by: <span className="font-medium text-[var(--color-text-primary)]">{job.uploaded_by_email}</span></p>
                    <p>Date: {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}</p>
                    <p>Dataset A: <span className="font-mono">{job.dataset_a_filename}</span></p>
                    <p>Dataset B: <span className="font-mono">{job.dataset_b_filename}</span></p>
                    {job.reviewed_by_email && <p>Reviewed by: {job.reviewed_by_email}</p>}
                    {job.rejection_reason && <p className="text-red-600">Reason: {job.rejection_reason}</p>}
                    {job.processing_error && <p className="text-red-600 col-span-2">Error: {job.processing_error}</p>}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2 shrink-0">
                  {/* Preview buttons always available */}
                  <button
                    onClick={() => handlePreview(job.id, 'a', job.dataset_a_filename)}
                    disabled={previewLoading === `${job.id}-a`}
                    className="px-3 py-1.5 text-xs font-medium text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 transition-colors inline-flex items-center gap-1.5"
                  >
                    {previewLoading === `${job.id}-a` && (
                      <div className="w-3 h-3 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
                    )}
                    Preview A
                  </button>
                  <button
                    onClick={() => handlePreview(job.id, 'b', job.dataset_b_filename)}
                    disabled={previewLoading === `${job.id}-b`}
                    className="px-3 py-1.5 text-xs font-medium text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 transition-colors inline-flex items-center gap-1.5"
                  >
                    {previewLoading === `${job.id}-b` && (
                      <div className="w-3 h-3 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
                    )}
                    Preview B
                  </button>

                  {job.status === 'pending_review' && (
                    <>
                      <button
                        onClick={() => handleApprove(job.id)}
                        disabled={actionLoading === job.id}
                        className="px-3 py-1.5 text-xs font-medium text-white bg-green-600 rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors"
                      >
                        {actionLoading === job.id ? '...' : 'Approve'}
                      </button>
                      <button
                        onClick={() => setRejectJobId(rejectJobId === job.id ? null : job.id)}
                        disabled={actionLoading === job.id}
                        className="px-3 py-1.5 text-xs font-medium text-white bg-red-600 rounded-md hover:bg-red-700 disabled:opacity-50 transition-colors"
                      >
                        Reject
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Inline reject reason input */}
              {rejectJobId === job.id && (
                <div className="mt-3 flex items-center gap-2 border-t border-gray-100 pt-3">
                  <input
                    type="text"
                    value={rejectReason}
                    onChange={(e) => setRejectReason(e.target.value)}
                    placeholder="Rejection reason (optional)"
                    className="flex-1 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-red-300"
                  />
                  <button
                    onClick={() => handleReject(job.id)}
                    disabled={actionLoading === job.id}
                    className="px-3 py-1.5 text-xs font-medium text-white bg-red-600 rounded-md hover:bg-red-700 disabled:opacity-50 transition-colors"
                  >
                    Confirm Reject
                  </button>
                  <button
                    onClick={() => { setRejectJobId(null); setRejectReason(''); }}
                    className="px-3 py-1.5 text-xs font-medium text-gray-600 hover:text-gray-800 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {preview && (
        <PreviewModal
          content={preview.content}
          filename={preview.filename}
          onClose={() => setPreview(null)}
        />
      )}
    </div>
  );
}
