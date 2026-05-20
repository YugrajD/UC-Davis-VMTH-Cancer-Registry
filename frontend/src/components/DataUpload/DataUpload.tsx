import { useState, useRef, useEffect, useCallback } from 'react';
import { uploadCSV, fetchMyJobs, type IngestionJob } from '../../api/client';
import { useAuth } from '../../contexts/AuthContext';
import { STAGE_LABELS } from '../shared/pipelineStages';
import { parseCsvPreview, type CsvPreview } from './csvPreview';

function PreviewModal({ file, onClose }: { file: File; onClose: () => void }) {
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        setPreview(parseCsvPreview(e.target?.result as string));
      } catch {
        setError('Could not parse file');
      }
    };
    reader.onerror = () => setError('Could not read file');
    reader.readAsText(file);
  }, [file]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/50" />
      <div
        className="relative bg-white rounded-xl shadow-2xl flex flex-col w-full max-w-5xl max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 shrink-0">
          <div>
            <h2 className="text-base font-semibold text-[var(--color-text-primary)]">{file.name}</h2>
            {preview && (
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                {preview.headers.length} columns &middot; {preview.totalRows.toLocaleString()} rows &middot;{' '}
                {(file.size / 1024).toFixed(1)} KB
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {error ? (
          <div className="p-6">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        ) : !preview ? (
          <div className="flex items-center justify-center py-16">
            <div className="w-6 h-6 border-2 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
          </div>
        ) : (
          <div className="overflow-auto flex-1 min-h-0">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 sticky top-0 z-10">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-400 w-12">#</th>
                  {preview.headers.map((h, i) => (
                    <th key={i} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {preview.rows.map((row, ri) => (
                  <tr key={ri} className="hover:bg-gray-50">
                    <td className="px-4 py-1.5 text-xs text-gray-400">{ri + 1}</td>
                    {row.map((cell, ci) => (
                      <td key={ci} className="px-4 py-1.5 text-gray-700 whitespace-nowrap max-w-[300px] truncate" title={cell}>
                        {cell || <span className="text-gray-300">&mdash;</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function FilePreview({ file }: { file: File }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="mt-2 text-xs text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium transition-colors"
      >
        Preview file
      </button>
      {open && <PreviewModal file={file} onClose={() => setOpen(false)} />}
    </>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending_review: 'bg-yellow-100 text-yellow-800',
    processing: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    rejected: 'bg-red-100 text-red-800',
  };
  const labels: Record<string, string> = {
    pending_review: 'Pending Review',
    processing: 'Processing',
    completed: 'Completed',
    failed: 'Failed',
    rejected: 'Rejected',
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${colors[status] || 'bg-gray-100 text-gray-800'}`}>
      {labels[status] || status}
    </span>
  );
}

export function DataUpload() {
  const { user, getAccessToken } = useAuth();
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [myJobs, setMyJobs] = useState<IngestionJob[]>([]);
  const [jobsLoading, setJobsLoading] = useState(false);

  const refA = useRef<HTMLInputElement>(null);
  const refB = useRef<HTMLInputElement>(null);

  const loadMyJobs = useCallback(async () => {
    const token = await getAccessToken();
    if (!token) return;
    setJobsLoading(true);
    try {
      const jobs = await fetchMyJobs(token);
      setMyJobs(jobs);
    } catch {
      // silently fail for job listing
    } finally {
      setJobsLoading(false);
    }
  }, [getAccessToken]);

  useEffect(() => {
    if (user) loadMyJobs();
  }, [user, loadMyJobs]);

  // Auto-poll every 10s while any job is processing
  useEffect(() => {
    const hasProcessing = myJobs.some(j => j.status === 'processing');
    if (!hasProcessing) return;
    const interval = setInterval(loadMyJobs, 10000);
    return () => clearInterval(interval);
  }, [myJobs, loadMyJobs]);

  const handleUpload = async () => {
    if (!fileA) {
      setError('Dataset A is required.');
      return;
    }

    setLoading(true);
    setError(null);
    setSubmitted(false);

    try {
      const token = await getAccessToken();
      await uploadCSV(fileA, fileB ?? undefined, token);
      setSubmitted(true);
      if (user) await loadMyJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFileA(null);
    setFileB(null);
    setSubmitted(false);
    setError(null);
    if (refA.current) refA.current.value = '';
    if (refB.current) refB.current.value = '';
  };

  return (
    <div className="space-y-6">
      {/* File Inputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Dataset A */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider mb-1">
            Dataset A — Clinical Notes
          </h3>
          <p className="text-xs text-[var(--color-text-secondary)] mb-4">
            CSV with columns:{' '}
            <code className="bg-gray-100 px-1 rounded">DtOfRq</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">Sex</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">Species</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">Breed</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">Diagnoses</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">Text</code>
          </p>
          <label className="block">
            <span className="sr-only">Choose Dataset A file</span>
            <input
              ref={refA}
              type="file"
              accept=".csv,.xlsx"
              onChange={(e) => setFileA(e.target.files?.[0] ?? null)}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-[var(--color-teal)] file:text-white
                hover:file:bg-[var(--color-teal-dark)] file:cursor-pointer"
            />
          </label>
          {fileA && (
            <>
              <p className="mt-2 text-xs text-green-700">{fileA.name} selected</p>
              <FilePreview file={fileA} />
            </>
          )}
        </div>

        {/* Dataset B */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider mb-1">
            Dataset B — Demographics
          </h3>
          <p className="text-xs text-[var(--color-text-secondary)] mb-4">
            CSV with columns:{' '}
            <code className="bg-gray-100 px-1 rounded">Sex</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">Zipcode</code>
          </p>
          <label className="block">
            <span className="sr-only">Choose Dataset B file</span>
            <input
              ref={refB}
              type="file"
              accept=".csv,.xlsx"
              onChange={(e) => setFileB(e.target.files?.[0] ?? null)}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-[var(--color-teal)] file:text-white
                hover:file:bg-[var(--color-teal-dark)] file:cursor-pointer"
            />
          </label>
          {fileB && (
            <>
              <p className="mt-2 text-xs text-green-700">{fileB.name} selected</p>
              <FilePreview file={fileB} />
            </>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleUpload}
          disabled={loading || !fileA}
          className="px-6 py-2.5 bg-[var(--color-teal)] text-white text-sm font-semibold rounded-md
            hover:bg-[var(--color-teal-dark)] disabled:opacity-50 disabled:cursor-not-allowed
            transition-colors"
        >
          {loading ? 'Submitting...' : 'Submit for Review'}
        </button>
        {(fileA || fileB || submitted) && (
          <button
            onClick={handleReset}
            disabled={loading}
            className="px-4 py-2.5 text-sm text-gray-600 hover:text-gray-900 transition-colors"
          >
            Reset
          </button>
        )}
      </div>

      {/* Loading indicator */}
      {loading && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <p className="text-sm text-blue-800">Uploading files for review...</p>
          <div className="mt-2 h-1.5 bg-blue-100 rounded-full overflow-hidden">
            <div className="h-full bg-blue-500 rounded-full animate-pulse w-2/3" />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm text-red-800 font-medium">Error</p>
          <p className="text-sm text-red-700 mt-1">{error}</p>
        </div>
      )}

      {/* Submitted confirmation */}
      {submitted && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <p className="text-sm text-green-800 font-medium">Submitted for Review</p>
          <p className="text-sm text-green-700 mt-1">
            Your upload has been submitted and is awaiting admin approval. You can track its status below.
          </p>
        </div>
      )}

      {/* My Uploads — only shown when signed in */}
      {user && <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-200 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">My Uploads</h3>
          <button
            onClick={loadMyJobs}
            className="text-xs text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium"
          >
            Refresh
          </button>
        </div>

        {jobsLoading ? (
          <div className="p-8 flex justify-center">
            <div className="w-6 h-6 border-2 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
          </div>
        ) : myJobs.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-sm text-[var(--color-text-secondary)]">No uploads yet.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Dataset A</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Dataset B</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Submitted</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {myJobs.map((job) => (
                  <tr key={job.id} className="hover:bg-gray-50">
                    <td className="px-4 py-2.5 text-gray-600">#{job.id}</td>
                    <td className="px-4 py-2.5 font-mono text-xs">{job.dataset_a_filename}</td>
                    <td className="px-4 py-2.5 font-mono text-xs">{job.dataset_b_filename}</td>
                    <td className="px-4 py-2.5">
                      <StatusBadge status={job.status} />
                      {job.status === 'processing' && job.processing_stage && (
                        <div className="flex items-center gap-1 mt-1">
                          <div className="w-2.5 h-2.5 border-2 border-gray-300 border-t-[var(--color-teal)] rounded-full animate-spin shrink-0" />
                          <span className="text-xs text-[var(--color-teal)]">
                            {STAGE_LABELS[job.processing_stage] ?? job.processing_stage}
                          </span>
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-xs text-gray-500">
                      {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>}
    </div>
  );
}
