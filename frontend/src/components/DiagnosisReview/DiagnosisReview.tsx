import { useCallback, useEffect, useMemo, useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import {
  fetchAllDiagnoses,
  fetchDiagnosisDetail,
  fetchPendingDiagnoses,
  reviewDiagnosis,
  type DiagnosisDetail,
  type PendingDiagnosis,
  type ReviewActionKind,
} from '../../api/client';

type StatusFilter = 'pending' | 'confirmed' | 'corrected' | 'rejected' | 'all';
const STATUS_FILTERS: StatusFilter[] = ['pending', 'confirmed', 'corrected', 'rejected', 'all'];

const PAGE_SIZE = 50;

function ConfidenceBar({ value }: { value: number | null }) {
  if (value === null) return <span className="text-xs text-gray-400">—</span>;
  const pct = Math.round(value * 100);
  const color =
    pct >= 80 ? 'bg-emerald-500' : pct >= 50 ? 'bg-amber-500' : 'bg-red-500';
  return (
    <div className="flex items-center gap-2 w-32">
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-600 w-8 tabular-nums">{pct}%</span>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const styles: Record<string, string> = {
    pending: 'bg-amber-100 text-amber-800 border-amber-200',
    confirmed: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    corrected: 'bg-blue-100 text-blue-800 border-blue-200',
    rejected: 'bg-red-100 text-red-800 border-red-200',
    flagged: 'bg-gray-100 text-gray-800 border-gray-200',
  };
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${
        styles[status] || styles.flagged
      }`}
    >
      {status}
    </span>
  );
}

function formatTs(iso: string): string {
  return new Date(iso).toLocaleString();
}

// Source text panel: truncates to ~3 lines / 240 chars by default with a
// "Show more / Show less" toggle.  Renders nothing when the text is null
// (legacy diagnoses ingested before the May 2026 fix don't have it).
const SOURCE_TEXT_TRUNCATE_AT = 240;

function SourceText({ text }: { text: string | null }) {
  const [expanded, setExpanded] = useState(false);

  if (!text || !text.trim()) {
    return (
      <div>
        <p className="text-xs text-gray-500 mb-1">Pathology report text</p>
        <p className="text-xs text-gray-400 italic">Pathology report text not available for this diagnosis.</p>
      </div>
    );
  }

  const trimmed = text.trim();
  const needsTruncation = trimmed.length > SOURCE_TEXT_TRUNCATE_AT;
  const shown = expanded || !needsTruncation
    ? trimmed
    : trimmed.slice(0, SOURCE_TEXT_TRUNCATE_AT).trimEnd() + '…';

  return (
    <div>
      <p className="text-xs text-gray-500 mb-1">Pathology report text</p>
      <p className="text-sm whitespace-pre-wrap text-[var(--color-text-primary)]">
        {shown}
      </p>
      {needsTruncation && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1 text-xs font-medium text-blue-600 hover:text-blue-800"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  );
}

interface DetailPanelProps {
  detail: DiagnosisDetail | null;
  loading: boolean;
  onAction: (action: ReviewActionKind, fields: { cancer_type_name?: string; icd_o_code?: string; notes?: string }) => Promise<void>;
  busy: boolean;
}

function DetailPanel({ detail, loading, onAction, busy }: DetailPanelProps) {
  if (loading) {
    return <div className="p-6 text-sm text-gray-500">Loading…</div>;
  }
  if (!detail) {
    return (
      <div className="p-6 text-sm text-gray-500">
        Select a diagnosis from the queue to begin review.
      </div>
    );
  }
  // Re-mount the body when the selected diagnosis changes so the form
  // fields reset to the new defaults without an effect-driven setState.
  return <DetailPanelBody key={detail.id} detail={detail} onAction={onAction} busy={busy} />;
}

interface DetailPanelBodyProps {
  detail: DiagnosisDetail;
  onAction: DetailPanelProps['onAction'];
  busy: boolean;
}

function DetailPanelBody({ detail, onAction, busy }: DetailPanelBodyProps) {
  const [correctName, setCorrectName] = useState(detail.cancer_type_name);
  const [correctIcd, setCorrectIcd] = useState(detail.icd_o_code ?? '');
  const [notes, setNotes] = useState('');

  return (
    <div className="p-6 space-y-5">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <h3 className="text-base font-semibold text-[var(--color-text-primary)]">
            {detail.cancer_type_name || 'Unknown'}
          </h3>
          <StatusPill status={detail.review_status} />
        </div>
        <p className="text-xs text-gray-500">
          Patient {detail.patient_anon_id ?? '—'} · diagnosis #{detail.diagnosis_index ?? '?'} · ICD-O {detail.icd_o_code ?? '—'}
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4 text-xs">
        <div>
          <p className="text-gray-500 mb-1">Confidence</p>
          <ConfidenceBar value={detail.confidence} />
        </div>
        <div>
          <p className="text-gray-500 mb-1">Top-1 vs Top-2</p>
          <p className="font-medium tabular-nums">
            {detail.top2_margin !== null ? `${(detail.top2_margin * 100).toFixed(0)}%` : '—'}
          </p>
        </div>
        <div>
          <p className="text-gray-500 mb-1">Method</p>
          <p className="font-medium">{detail.prediction_method ?? '—'}</p>
        </div>
      </div>

      <SourceText text={detail.original_text} />

      {detail.predicted_term && (
        <div>
          <p className="text-xs text-gray-500 mb-1">Predicted term</p>
          <p className="text-sm">{detail.predicted_term}</p>
        </div>
      )}

      {detail.original_predicted_term && (
        <div className="bg-gray-50 border border-gray-200 rounded p-3">
          <p className="text-xs text-gray-500 mb-1">Original PetBERT prediction (before correction)</p>
          <p className="text-sm">
            {detail.original_predicted_term} · ICD-O {detail.original_icd_o_code ?? '—'}
          </p>
        </div>
      )}

      {detail.review_status !== 'rejected' && (
        <div className="border-t border-gray-200 pt-4 space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-700">
            Review action
          </h4>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-gray-600">
              Cancer type (for correction)
              <input
                type="text"
                value={correctName}
                onChange={(e) => setCorrectName(e.target.value)}
                className="mt-1 w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
              />
            </label>
            <label className="text-xs text-gray-600">
              ICD-O code
              <input
                type="text"
                value={correctIcd}
                onChange={(e) => setCorrectIcd(e.target.value)}
                className="mt-1 w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
              />
            </label>
          </div>
          <label className="text-xs text-gray-600 block">
            Notes (optional)
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
              className="mt-1 w-full px-2 py-1.5 text-sm border border-gray-300 rounded resize-none"
            />
          </label>

          <div className="flex gap-2">
            <button
              onClick={() => onAction('confirm', { notes: notes || undefined })}
              disabled={busy}
              className="px-3 py-1.5 text-sm font-medium bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50"
            >
              Confirm
            </button>
            <button
              onClick={() =>
                onAction('correct', {
                  cancer_type_name: correctName.trim(),
                  icd_o_code: correctIcd.trim() || undefined,
                  notes: notes || undefined,
                })
              }
              disabled={busy || !correctName.trim() || correctName === detail.cancer_type_name}
              className="px-3 py-1.5 text-sm font-medium bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              Correct
            </button>
            <button
              onClick={() => onAction('reject', { notes: notes || undefined })}
              disabled={busy}
              className="px-3 py-1.5 text-sm font-medium bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
            >
              Reject
            </button>
          </div>
        </div>
      )}

      <div className="border-t border-gray-200 pt-4">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-700 mb-2">
          History ({detail.events.length})
        </h4>
        <ul className="space-y-2">
          {detail.events.map((e) => (
            <li key={e.id} className="text-xs text-gray-600 flex gap-2">
              <span className="text-gray-400 tabular-nums whitespace-nowrap">
                {formatTs(e.created_at)}
              </span>
              <span className="font-medium">{e.actor_email}</span>
              <span>{e.action}</span>
              {e.from_status && (
                <span className="text-gray-400">
                  {e.from_status} → {e.to_status}
                </span>
              )}
              {e.notes && <span className="italic text-gray-500">"{e.notes}"</span>}
            </li>
          ))}
          {detail.events.length === 0 && (
            <li className="text-xs text-gray-400">No history yet.</li>
          )}
        </ul>
      </div>
    </div>
  );
}

export function DiagnosisReview() {
  const { getAccessToken, isUploader, isAdmin } = useAuth();
  const canAudit = isUploader || isAdmin;
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('pending');
  const [pending, setPending] = useState<PendingDiagnosis[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<DiagnosisDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [busy, setBusy] = useState(false);
  const [page, setPage] = useState(0);

  const load = useCallback(async () => {
    const token = await getAccessToken();
    if (!token) return;
    setLoadingList(true);
    setError(null);
    try {
      let rows: PendingDiagnosis[];
      if (canAudit && statusFilter !== 'pending') {
        rows = await fetchAllDiagnoses(token, {
          status: statusFilter === 'all' ? undefined : statusFilter,
          limit: PAGE_SIZE,
          offset: page * PAGE_SIZE,
        });
      } else {
        rows = await fetchPendingDiagnoses(token, { limit: PAGE_SIZE, offset: page * PAGE_SIZE });
      }
      setPending(rows);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    } finally {
      setLoadingList(false);
    }
  }, [getAccessToken, page, canAudit, statusFilter]);

  useEffect(() => {
    load(); // eslint-disable-line react-hooks/set-state-in-effect
  }, [load]);

  const loadDetail = useCallback(
    async (id: number) => {
      const token = await getAccessToken();
      if (!token) return;
      setLoadingDetail(true);
      try {
        const d = await fetchDiagnosisDetail(token, id);
        setDetail(d);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load detail');
      } finally {
        setLoadingDetail(false);
      }
    },
    [getAccessToken],
  );

  useEffect(() => {
    if (selectedId !== null) loadDetail(selectedId); // eslint-disable-line react-hooks/set-state-in-effect
    else setDetail(null);
  }, [selectedId, loadDetail]);

  const handleAction = useCallback(
    async (
      action: ReviewActionKind,
      fields: { cancer_type_name?: string; icd_o_code?: string; notes?: string },
    ) => {
      if (selectedId === null) return;
      const token = await getAccessToken();
      if (!token) return;
      setBusy(true);
      setError(null);
      try {
        await reviewDiagnosis(token, selectedId, { action, ...fields });
        // Remove from queue and clear selection so the user sees they made progress.
        setPending((rows) => rows.filter((r) => r.id !== selectedId));
        setSelectedId(null);
        setDetail(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Action failed');
      } finally {
        setBusy(false);
      }
    },
    [getAccessToken, selectedId],
  );

  const handleFilterChange = useCallback((f: StatusFilter) => {
    setStatusFilter(f);
    setPage(0);
    setSelectedId(null);
    setDetail(null);
  }, []);

  const headerCount = useMemo(() => pending.length, [pending]);

  // Group diagnoses by ingestion_job_id, preserving server sort order within
  // each group.  The key is the job id (or -1 for legacy/unlinked rows).
  const grouped = useMemo(() => {
    const map = new Map<number, { jobId: number | null; filename: string | null; createdAt: string | null; items: PendingDiagnosis[] }>();
    for (const d of pending) {
      const key = d.ingestion_job_id ?? -1;
      let group = map.get(key);
      if (!group) {
        group = { jobId: d.ingestion_job_id, filename: d.job_filename, createdAt: d.job_created_at, items: [] };
        map.set(key, group);
      }
      group.items.push(d);
    }
    return Array.from(map.values());
  }, [pending]);

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">
              Diagnosis Review
            </h2>
            <p className="text-sm text-[var(--color-text-secondary)] mt-1">
              Triage low-confidence and ambiguous predictions before they enter
              public dashboard stats. Confirm to accept, Correct to assign a
              different cancer type, or Reject to drop the prediction.
            </p>
          </div>
          {canAudit && (
            <div className="flex gap-1 shrink-0 flex-wrap justify-end">
              {STATUS_FILTERS.map((f) => (
                <button
                  key={f}
                  onClick={() => handleFilterChange(f)}
                  className={`px-3 py-1 text-xs rounded-full font-medium border transition-colors ${
                    statusFilter === f
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {f.charAt(0).toUpperCase() + f.slice(1)}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-7 bg-white rounded-lg border border-gray-200">
          <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
            <span className="text-sm font-medium">
              Pending {headerCount > 0 && <span className="text-gray-500">({headerCount})</span>}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0 || loadingList}
                className="px-2 py-1 text-xs text-gray-600 disabled:opacity-40"
              >
                Prev
              </button>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={pending.length < PAGE_SIZE || loadingList}
                className="px-2 py-1 text-xs text-gray-600 disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>
          {loadingList ? (
            <div className="p-6 text-sm text-gray-500">Loading...</div>
          ) : pending.length === 0 ? (
            <div className="p-6 text-sm text-gray-500">
              {page === 0
                ? 'No diagnoses awaiting review.'
                : 'No more rows on this page.'}
            </div>
          ) : (
            <div>
              {grouped.map((group) => (
                <div key={group.jobId ?? 'unlinked'}>
                  <div className="sticky top-0 z-10 bg-gray-50 border-b border-gray-200 px-4 py-2">
                    <span className="text-xs font-semibold text-gray-700">
                      {group.jobId != null
                        ? `Job #${group.jobId}`
                        : 'Unlinked'}
                    </span>
                    {group.filename && (
                      <span className="text-xs text-gray-500 ml-2">
                        {group.filename}
                      </span>
                    )}
                    {group.createdAt && (
                      <span className="text-xs text-gray-400 ml-2">
                        {formatTs(group.createdAt)}
                      </span>
                    )}
                    <span className="text-xs text-gray-400 ml-2">
                      ({group.items.length})
                    </span>
                  </div>
                  <ul className="divide-y divide-gray-100">
                    {group.items.map((d) => {
                      const active = d.id === selectedId;
                      return (
                        <li
                          key={d.id}
                          onClick={() => setSelectedId(d.id)}
                          className={`px-4 py-3 cursor-pointer hover:bg-gray-50 ${
                            active ? 'bg-blue-50' : ''
                          }`}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0 flex-1">
                              <p className="text-sm font-medium text-gray-900 truncate">
                                {d.cancer_type_name}
                              </p>
                              <p className="text-xs text-gray-500 truncate">
                                {d.patient_anon_id ?? '—'} · {d.predicted_term ?? d.icd_o_code ?? '—'}
                              </p>
                            </div>
                            <ConfidenceBar value={d.confidence} />
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="col-span-5 bg-white rounded-lg border border-gray-200">
          <DetailPanel
            detail={detail}
            loading={loadingDetail}
            onAction={handleAction}
            busy={busy}
          />
        </div>
      </div>
    </div>
  );
}
