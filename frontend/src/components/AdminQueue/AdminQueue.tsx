import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { fetchJobs, reviewJob, fetchJobPreview, cancelJob, fetchAvailableModels, type IngestionJob } from '../../api/client';
import { STAGE_LABELS, LOCAL_STAGES, GCP_STAGES } from '../shared/pipelineStages';

const ACTIVE_STATUSES = ['pending_review', 'processing'];
const ARCHIVE_STATUSES = ['completed', 'failed', 'rejected', 'cancelled'];

function PipelineStageIndicator({ stage }: { stage: string }) {
  const isGcp = GCP_STAGES.includes(stage);
  const steps = isGcp ? GCP_STAGES : LOCAL_STAGES;
  const currentIndex = steps.indexOf(stage);

  return (
    <div className="mt-3 pt-3 border-t border-gray-100">
      <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">Pipeline stage</p>
      <ol className="flex flex-wrap gap-y-2 gap-x-0">
        {steps.map((s, i) => {
          const isDone    = i < currentIndex;
          const isActive  = i === currentIndex;
          const isPending = i > currentIndex;
          return (
            <li key={s} className="flex items-center">
              {/* Step dot */}
              <div className={`flex items-center justify-center w-5 h-5 rounded-full shrink-0 text-xs font-bold
                ${isDone    ? 'bg-green-500 text-white' : ''}
                ${isActive  ? 'bg-[var(--color-teal)] text-white ring-2 ring-[var(--color-teal)] ring-offset-1' : ''}
                ${isPending ? 'bg-gray-200 text-gray-400' : ''}
              `}>
                {isDone ? (
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  <span>{i + 1}</span>
                )}
              </div>
              {/* Step label */}
              <span className={`ml-1.5 text-xs whitespace-nowrap
                ${isDone    ? 'text-green-600' : ''}
                ${isActive  ? 'text-[var(--color-teal)] font-semibold' : ''}
                ${isPending ? 'text-gray-400' : ''}
              `}>
                {STAGE_LABELS[s] ?? s}
              </span>
              {/* Connector line */}
              {i < steps.length - 1 && (
                <span className={`mx-2 text-xs ${isDone ? 'text-green-400' : 'text-gray-300'}`}>›</span>
              )}
            </li>
          );
        })}
      </ol>
      {/* Spinner next to active label */}
      <div className="flex items-center gap-1.5 mt-2">
        <div className="w-3 h-3 border-2 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
        <span className="text-xs text-[var(--color-teal)]">{STAGE_LABELS[stage] ?? stage}…</span>
      </div>
    </div>
  );
}

type ArchiveFilter = 'all' | 'completed' | 'failed' | 'rejected' | 'cancelled';

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending_review: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    processing: 'bg-blue-100 text-blue-800 border-blue-200',
    completed: 'bg-green-100 text-green-800 border-green-200',
    failed: 'bg-red-100 text-red-800 border-red-200',
    rejected: 'bg-red-100 text-red-800 border-red-200',
    cancelled: 'bg-gray-100 text-gray-600 border-gray-200',
  };
  const labels: Record<string, string> = {
    pending_review: 'Pending Review',
    processing: 'Processing',
    completed: 'Completed',
    failed: 'Failed',
    rejected: 'Rejected',
    cancelled: 'Cancelled',
  };

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${colors[status] || 'bg-gray-100 text-gray-800 border-gray-200'}`}>
      {labels[status] || status}
    </span>
  );
}

function ResultSummary({ summary }: { summary: NonNullable<IngestionJob['result_summary']> }) {
  const total = summary.high_confidence + summary.medium_confidence + summary.low_confidence;
  const pct = (n: number) => total > 0 ? Math.round((n / total) * 100) : 0;

  return (
    <div className="mt-3 pt-3 border-t border-gray-100">
      <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">Model results</p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
        <div className="bg-gray-50 rounded-md px-3 py-2 text-center">
          <p className="text-lg font-bold text-[var(--color-text-primary)]">{summary.patients}</p>
          <p className="text-xs text-[var(--color-text-secondary)]">Records</p>
        </div>
        <div className="bg-gray-50 rounded-md px-3 py-2 text-center">
          <p className="text-lg font-bold text-[var(--color-text-primary)]">{summary.diagnoses}</p>
          <p className="text-xs text-[var(--color-text-secondary)]">Diagnoses</p>
        </div>
        <div className="bg-gray-50 rounded-md px-3 py-2 text-center">
          <p className="text-lg font-bold text-[var(--color-text-primary)]">
            {summary.avg_confidence !== null ? `${summary.avg_confidence}%` : '—'}
          </p>
          <p className="text-xs text-[var(--color-text-secondary)]">Avg confidence</p>
        </div>
        <div className="bg-gray-50 rounded-md px-3 py-2">
          <div className="flex items-center gap-1 mb-1">
            <span className="w-2 h-2 rounded-full bg-green-500 shrink-0" />
            <span className="text-xs text-gray-600">{pct(summary.high_confidence)}% high ≥80%</span>
          </div>
          <div className="flex items-center gap-1 mb-1">
            <span className="w-2 h-2 rounded-full bg-yellow-400 shrink-0" />
            <span className="text-xs text-gray-600">{pct(summary.medium_confidence)}% med 50–79%</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-red-400 shrink-0" />
            <span className="text-xs text-gray-600">{pct(summary.low_confidence)}% low &lt;50%</span>
          </div>
        </div>
      </div>
      {(summary.top_cancer_types ?? []).length > 0 && (
        <div>
          <p className="text-xs text-[var(--color-text-secondary)] mb-1">Top predicted cancer types</p>
          <div className="flex flex-wrap gap-1.5">
            {(summary.top_cancer_types ?? []).map(ct => (
              <span key={ct.name} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-teal-50 text-teal-800 text-xs border border-teal-100">
                {ct.name}
                <span className="font-semibold">{ct.count}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
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

function JobCard({
  job,
  actionLoading,
  previewLoading,
  approveJobId,
  selectedModelFolder,
  availableModels,
  rejectJobId,
  rejectReason,
  onApproveOpen,
  onApproveConfirm,
  onApproveCancel,
  onModelFolderChange,
  onRejectOpen,
  onRejectConfirm,
  onRejectCancel,
  onRejectReasonChange,
  onPreview,
  onCancel,
}: {
  job: IngestionJob;
  actionLoading: number | null;
  previewLoading: string | null;
  approveJobId: number | null;
  selectedModelFolder: string;
  availableModels: string[];
  rejectJobId: number | null;
  rejectReason: string;
  onApproveOpen: (id: number) => void;
  onApproveConfirm: (id: number) => void;
  onApproveCancel: () => void;
  onModelFolderChange: (v: string) => void;
  onRejectOpen: (id: number) => void;
  onRejectConfirm: (id: number) => void;
  onRejectCancel: () => void;
  onRejectReasonChange: (v: string) => void;
  onPreview: (id: number, filename: string) => void;
  onCancel: (id: number) => void;
}) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5">
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
            <p>File: <span className="font-mono">{job.dataset_a_filename}</span></p>
            {job.model_folder && <p>Model: <span className="font-medium text-[var(--color-text-primary)]">{job.model_folder}</span></p>}
            {job.reviewed_by_email && <p>Reviewed by: {job.reviewed_by_email}</p>}
            {job.reviewed_at && <p>Reviewed: {new Date(job.reviewed_at).toLocaleString()}</p>}
            {job.rejection_reason && <p className="text-red-600">Reason: {job.rejection_reason}</p>}
            {job.processing_error && <p className="text-red-600 col-span-2">Error: {job.processing_error}</p>}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {['completed', 'failed'].includes(job.status) ? (
            <span
              title="File was removed after processing"
              className="px-3 py-1.5 text-xs font-medium text-gray-400 border border-gray-200 rounded-md cursor-not-allowed inline-flex items-center gap-1.5"
            >
              Preview
            </span>
          ) : (
            <button
              onClick={() => onPreview(job.id, job.dataset_a_filename)}
              disabled={previewLoading === `${job.id}`}
              className="px-3 py-1.5 text-xs font-medium text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 transition-colors inline-flex items-center gap-1.5"
            >
              {previewLoading === `${job.id}` && (
                <div className="w-3 h-3 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
              )}
              Preview
            </button>
          )}

          {job.status === 'pending_review' && (
            <>
              <button
                onClick={() => onApproveOpen(job.id)}
                disabled={actionLoading === job.id}
                className="px-3 py-1.5 text-xs font-medium text-white bg-green-600 rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                Approve
              </button>
              <button
                onClick={() => onRejectOpen(job.id)}
                disabled={actionLoading === job.id}
                className="px-3 py-1.5 text-xs font-medium text-white bg-red-600 rounded-md hover:bg-red-700 disabled:opacity-50 transition-colors"
              >
                Reject
              </button>
            </>
          )}
          {job.status === 'processing' && (
            <button
              onClick={() => onCancel(job.id)}
              disabled={actionLoading === job.id}
              className="px-3 py-1.5 text-xs font-medium text-white bg-orange-500 rounded-md hover:bg-orange-600 disabled:opacity-50 transition-colors"
            >
              {actionLoading === job.id ? '...' : 'Cancel'}
            </button>
          )}
        </div>
      </div>

      {/* Pipeline stage indicator — only shown while processing */}
      {job.status === 'processing' && job.processing_stage && (
        <PipelineStageIndicator stage={job.processing_stage} />
      )}

      {/* Model result summary — shown on completed jobs */}
      {job.status === 'completed' && job.result_summary && (
        <ResultSummary summary={job.result_summary} />
      )}

      {/* Approve confirmation panel with model selector */}
      {approveJobId === job.id && (
        <div className="mt-3 border-t border-gray-100 pt-3 space-y-2">
          <p className="text-xs font-medium text-[var(--color-text-secondary)]">Select model</p>
          <div className="flex items-center gap-2">
            <select
              value={selectedModelFolder}
              onChange={(e) => onModelFolderChange(e.target.value)}
              disabled={actionLoading === job.id}
              className="flex-1 px-3 py-1.5 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-green-300 disabled:opacity-50"
            >
              {availableModels.map(folder => (
                <option key={folder} value={folder}>{folder}</option>
              ))}
            </select>
            <button
              onClick={() => onApproveConfirm(job.id)}
              disabled={actionLoading === job.id}
              className="px-3 py-1.5 text-xs font-medium text-white bg-green-600 rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors"
            >
              {actionLoading === job.id ? '...' : 'Confirm Approve'}
            </button>
            <button
              onClick={onApproveCancel}
              className="px-3 py-1.5 text-xs font-medium text-gray-600 hover:text-gray-800 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Reject confirmation panel */}
      {rejectJobId === job.id && (
        <div className="mt-3 flex items-center gap-2 border-t border-gray-100 pt-3">
          <input
            type="text"
            value={rejectReason}
            onChange={(e) => onRejectReasonChange(e.target.value)}
            placeholder="Rejection reason (optional)"
            className="flex-1 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-red-300"
          />
          <button
            onClick={() => onRejectConfirm(job.id)}
            disabled={actionLoading === job.id}
            className="px-3 py-1.5 text-xs font-medium text-white bg-red-600 rounded-md hover:bg-red-700 disabled:opacity-50 transition-colors"
          >
            Confirm Reject
          </button>
          <button
            onClick={onRejectCancel}
            className="px-3 py-1.5 text-xs font-medium text-gray-600 hover:text-gray-800 transition-colors"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}

export function AdminQueue() {
  const { getAccessToken } = useAuth();
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [approveJobId, setApproveJobId] = useState<number | null>(null);
  const [selectedModelFolder, setSelectedModelFolder] = useState('production');
  const [availableModels, setAvailableModels] = useState<string[]>(['production']);
  const [rejectJobId, setRejectJobId] = useState<number | null>(null);
  const [rejectReason, setRejectReason] = useState('');
  const [preview, setPreview] = useState<{ content: string; filename: string } | null>(null);
  const [previewLoading, setPreviewLoading] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<'queue' | 'archive'>('queue');
  const [archiveFilter, setArchiveFilter] = useState<ArchiveFilter>('all');
  const [archivePage, setArchivePage] = useState(1);
  const ARCHIVE_PAGE_SIZE = 10;

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
    loadJobs(); // eslint-disable-line react-hooks/set-state-in-effect
  }, [loadJobs]);

  // Fetch available model folders once on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const token = await getAccessToken();
        if (!token || cancelled) return;
        const models = await fetchAvailableModels(token);
        if (!cancelled) {
          setAvailableModels(models.length > 0 ? models : ['production']);
          setSelectedModelFolder(models.includes('production') ? 'production' : (models[0] ?? 'production'));
        }
      } catch {
        // keep the default ['production'] on error
      }
    })();
    return () => { cancelled = true; };
  }, [getAccessToken]);

  // Auto-poll when any job is processing
  useEffect(() => {
    const hasProcessing = jobs.some(j => j.status === 'processing');
    if (!hasProcessing) return;
    const interval = setInterval(loadJobs, 10000);
    return () => clearInterval(interval);
  }, [jobs, loadJobs]);

  const handleApproveOpen = (jobId: number) => {
    setRejectJobId(null);
    setRejectReason('');
    setApproveJobId(approveJobId === jobId ? null : jobId);
  };

  const handleApproveConfirm = async (jobId: number) => {
    setActionLoading(jobId);
    try {
      const token = await getAccessToken();
      if (!token) return;
      await reviewJob(token, jobId, 'approve', undefined, selectedModelFolder);
      setApproveJobId(null);
      await loadJobs();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Approve failed');
    } finally {
      setActionLoading(null);
    }
  };

  const handleApproveCancel = () => setApproveJobId(null);

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

  const handleCancel = async (jobId: number) => {
    if (!confirm('Cancel this job? This will stop processing and cannot be undone.')) return;
    setActionLoading(jobId);
    try {
      const token = await getAccessToken();
      if (!token) return;
      await cancelJob(token, jobId);
      await loadJobs();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Cancel failed');
    } finally {
      setActionLoading(null);
    }
  };

  const handlePreview = async (jobId: number, filename: string) => {
    const key = `${jobId}`;
    setPreviewLoading(key);
    try {
      const token = await getAccessToken();
      if (!token) return;
      const content = await fetchJobPreview(token, jobId);
      setPreview({ content, filename });
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Preview failed');
    } finally {
      setPreviewLoading(null);
    }
  };

  const queueJobs = jobs.filter(j => ACTIVE_STATUSES.includes(j.status));
  const allArchiveJobs = jobs
    .filter(j => ARCHIVE_STATUSES.includes(j.status))
    .filter(j => archiveFilter === 'all' || j.status === archiveFilter);
  const archiveTotalPages = Math.max(1, Math.ceil(allArchiveJobs.length / ARCHIVE_PAGE_SIZE));
  const archiveJobs = allArchiveJobs.slice(
    (archivePage - 1) * ARCHIVE_PAGE_SIZE,
    archivePage * ARCHIVE_PAGE_SIZE,
  );

  const sharedJobCardProps = {
    actionLoading,
    previewLoading,
    approveJobId,
    selectedModelFolder,
    availableModels,
    rejectJobId,
    rejectReason,
    onApproveOpen: handleApproveOpen,
    onApproveConfirm: handleApproveConfirm,
    onApproveCancel: handleApproveCancel,
    onModelFolderChange: setSelectedModelFolder,
    onRejectOpen: (id: number) => { setApproveJobId(null); setRejectJobId(rejectJobId === id ? null : id); },
    onRejectConfirm: handleReject,
    onRejectCancel: () => { setRejectJobId(null); setRejectReason(''); },
    onRejectReasonChange: setRejectReason,
    onPreview: handlePreview,
    onCancel: handleCancel,
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
      {/* Header */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setActiveView('queue')}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
                activeView === 'queue'
                  ? 'bg-white text-[var(--color-text-primary)] shadow-sm'
                  : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
              }`}
            >
              Queue
              {queueJobs.length > 0 && (
                <span className={`ml-2 px-1.5 py-0.5 rounded-full text-xs font-semibold ${
                  activeView === 'queue' ? 'bg-[var(--color-teal)] text-white' : 'bg-gray-300 text-gray-700'
                }`}>
                  {queueJobs.length}
                </span>
              )}
            </button>
            <button
              onClick={() => setActiveView('archive')}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
                activeView === 'archive'
                  ? 'bg-white text-[var(--color-text-primary)] shadow-sm'
                  : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
              }`}
            >
              Archive
              {ARCHIVE_STATUSES.some(s => jobs.some(j => j.status === s)) && (
                <span className={`ml-2 px-1.5 py-0.5 rounded-full text-xs font-semibold ${
                  activeView === 'archive' ? 'bg-gray-600 text-white' : 'bg-gray-300 text-gray-700'
                }`}>
                  {jobs.filter(j => ARCHIVE_STATUSES.includes(j.status)).length}
                </span>
              )}
            </button>
          </div>

          <div className="flex items-center gap-3">
            {activeView === 'archive' && (
              <select
                value={archiveFilter}
                onChange={e => { setArchiveFilter(e.target.value as ArchiveFilter); setArchivePage(1); }}
                className="text-sm border border-gray-300 rounded-md px-2 py-1.5 text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] bg-white"
              >
                <option value="all">All statuses</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="rejected">Rejected</option>
                <option value="cancelled">Cancelled</option>
              </select>
            )}
            <button
              onClick={loadJobs}
              className="text-sm text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Queue View */}
      {activeView === 'queue' && (
        queueJobs.length === 0 ? (
          <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
            <p className="text-sm text-[var(--color-text-secondary)]">No active jobs in the queue.</p>
            <p className="text-xs text-gray-400 mt-1">Completed, failed, and rejected jobs are in the Archive.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {queueJobs.map(job => (
              <JobCard key={job.id} job={job} {...sharedJobCardProps} />
            ))}
          </div>
        )
      )}

      {/* Archive View */}
      {activeView === 'archive' && (
        allArchiveJobs.length === 0 ? (
          <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
            <p className="text-sm text-[var(--color-text-secondary)]">
              {archiveFilter === 'all'
                ? 'No archived jobs yet.'
                : `No ${archiveFilter} jobs found.`}
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {archiveJobs.map(job => (
              <JobCard key={job.id} job={job} {...sharedJobCardProps} />
            ))}
            {/* Pagination controls */}
            <div className="bg-white rounded-lg border border-gray-200 px-4 py-3 flex items-center justify-between">
              <p className="text-xs text-[var(--color-text-secondary)]">
                Showing {(archivePage - 1) * ARCHIVE_PAGE_SIZE + 1}–{Math.min(archivePage * ARCHIVE_PAGE_SIZE, allArchiveJobs.length)} of {allArchiveJobs.length} jobs
              </p>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setArchivePage(p => Math.max(1, p - 1))}
                  disabled={archivePage === 1}
                  className="px-2.5 py-1 text-xs font-medium text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  ‹ Prev
                </button>
                {Array.from({ length: archiveTotalPages }, (_, i) => i + 1).map(page => (
                  <button
                    key={page}
                    onClick={() => setArchivePage(page)}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
                      page === archivePage
                        ? 'bg-[var(--color-teal)] text-white'
                        : 'text-gray-600 border border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    {page}
                  </button>
                ))}
                <button
                  onClick={() => setArchivePage(p => Math.min(archiveTotalPages, p + 1))}
                  disabled={archivePage === archiveTotalPages}
                  className="px-2.5 py-1 text-xs font-medium text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  Next ›
                </button>
              </div>
            </div>
          </div>
        )
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
