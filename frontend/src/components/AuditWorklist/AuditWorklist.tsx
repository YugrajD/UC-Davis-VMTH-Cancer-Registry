import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useLocalStorageState } from '../../hooks/useLocalStorageState';
import { HIDE_PREDICTIONS_BY_DEFAULT_KEY } from '../../lib/auditWorklistPrefs';
import {
  fetchAuditWorklist,
  fetchTaxonomyTerms,
  fetchAuditCaseDetail,
  saveCaseReview,
  reopenCaseReview,
  importAuditList,
  createGoldExport,
  downloadGoldExport,
  type WorklistCase,
  type WorklistResponse,
  type TaxonomyTermOut,
  type AuditCaseDetail,
  type GoldExportSummary,
} from '../../api/client';
import { CodePicker } from './CodePicker';

function friendlyError(e: unknown, fallback: string): string {
  return e instanceof Error && e.message ? e.message : fallback;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function formatTs(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

const CODE_SOURCE_LABELS: Record<string, string> = {
  manual: 'a specialist',
  diagnosis: 'diagnosis text',
  report: 'the report model',
};

function StatusBadge({ status }: { status: WorklistCase['review_status'] }) {
  const styles: Record<WorklistCase['review_status'], string> = {
    unreviewed: 'bg-gray-100 text-gray-600 border-gray-200',
    reviewed: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    locked: 'bg-blue-100 text-blue-800 border-blue-200',
  };
  const labels: Record<WorklistCase['review_status'], string> = {
    unreviewed: 'Unreviewed',
    reviewed: 'Reviewed',
    locked: 'Exported',
  };
  return (
    <span className={`shrink-0 inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-medium border ${styles[status]}`}>
      {labels[status]}
    </span>
  );
}

interface CodeIn {
  taxonomy_group: string;
  taxonomy_term: string;
}

const WORKLIST_PAGE_SIZE = 10;

export function AuditWorklist() {
  const { getAccessToken, isAdmin } = useAuth();

  const [worklist, setWorklist] = useState<WorklistResponse | null>(null);
  const [loadingWorklist, setLoadingWorklist] = useState(true);
  const [worklistError, setWorklistError] = useState<string | null>(null);

  const [terms, setTerms] = useState<TaxonomyTermOut[]>([]);

  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [detail, setDetail] = useState<AuditCaseDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Reviewer preference, set on the Settings page: whether a case with no
  // prior review opens with the model's predictions already visible, or
  // hidden until the reviewer asks for them (closer to a blind review).
  const [hidePredictionsByDefault] = useLocalStorageState(HIDE_PREDICTIONS_BY_DEFAULT_KEY, false);
  const [predictionsRevealed, setPredictionsRevealed] = useState(true);

  const [noCancer, setNoCancer] = useState(false);
  const [codes, setCodes] = useState<CodeIn[]>([]);
  // The pre-filled baseline (from an existing review, or from the registry's
  // current combined-predictions codes when there's no review yet) — used
  // only to tell "Approve" (submitting this unchanged) from "Save correction"
  // (the reviewer edited it) in the button label. Never sent to the server.
  const [baselineNoCancer, setBaselineNoCancer] = useState(false);
  const [baselineCodes, setBaselineCodes] = useState<CodeIn[]>([]);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const [showAdminTools, setShowAdminTools] = useState(false);
  const [importingList, setImportingList] = useState(false);
  const [importSummary, setImportSummary] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const listFileRef = useRef<HTMLInputElement>(null);
  const manifestFileRef = useRef<HTMLInputElement>(null);

  const [exportEmail, setExportEmail] = useState('');
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<GoldExportSummary | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  const [worklistPage, setWorklistPage] = useState(0);

  // Reset to page 1 whenever a different audit list is loaded (e.g. a new
  // import replaces the active list) so the user isn't left stranded on a
  // page number that may no longer exist.
  useEffect(() => {
    setWorklistPage(0);
  }, [worklist?.list_id]);

  const totalWorklistPages = Math.max(1, Math.ceil((worklist?.cases.length ?? 0) / WORKLIST_PAGE_SIZE));
  const currentWorklistPage = Math.min(worklistPage, totalWorklistPages - 1);
  const pagedCases = useMemo(
    () => worklist?.cases.slice(
      currentWorklistPage * WORKLIST_PAGE_SIZE,
      currentWorklistPage * WORKLIST_PAGE_SIZE + WORKLIST_PAGE_SIZE,
    ) ?? [],
    [worklist, currentWorklistPage],
  );

  const loadWorklist = useCallback(async () => {
    const token = await getAccessToken();
    if (!token) return;
    setLoadingWorklist(true);
    setWorklistError(null);
    try {
      setWorklist(await fetchAuditWorklist(token));
    } catch (e) {
      setWorklistError(friendlyError(e, 'Failed to load worklist'));
    } finally {
      setLoadingWorklist(false);
    }
  }, [getAccessToken]);

  useEffect(() => {
    loadWorklist();
  }, [loadWorklist]);

  useEffect(() => {
    (async () => {
      const token = await getAccessToken();
      if (!token) return;
      try {
        setTerms(await fetchTaxonomyTerms(token));
      } catch {
        // The picker just stays empty — not fatal to viewing the worklist.
      }
    })();
  }, [getAccessToken]);

  // Fetches and sets detail only — no side effects on the save/error messages,
  // so a post-save/post-reopen refresh doesn't wipe the confirmation it's
  // meant to follow. selectCase (below) is the version for user navigation,
  // which does clear those messages.
  const loadCaseDetail = useCallback(
    async (caseId: string) => {
      setDetail(null);
      setDetailError(null);
      const token = await getAccessToken();
      if (!token) return;
      setLoadingDetail(true);
      try {
        setDetail(await fetchAuditCaseDetail(token, caseId));
      } catch (e) {
        setDetailError(friendlyError(e, 'Failed to load case'));
      } finally {
        setLoadingDetail(false);
      }
    },
    [getAccessToken],
  );

  const selectCase = useCallback(
    async (caseId: string) => {
      setSelectedCaseId(caseId);
      setSaveError(null);
      setSaveSuccess(false);
      await loadCaseDetail(caseId);
    },
    [loadCaseDetail],
  );

  useEffect(() => {
    if (!detail) {
      setNoCancer(false);
      setCodes([]);
      setBaselineNoCancer(false);
      setBaselineCodes([]);
      setPredictionsRevealed(true);
      return;
    }
    // A prior review takes precedence (editing the specialist's own earlier
    // answer) and is always shown — the hide-by-default setting only applies
    // to the *model's* predictions, not a human reviewer's own past answer.
    // Otherwise, pre-fill from the registry's current codes — the reviewer
    // approves them as-is or corrects them, rather than starting blank. ML
    // has accepted this makes review non-blind for the evaluation batch;
    // that's a deliberate tradeoff on their side. The hide-by-default setting
    // lets a reviewer opt out of that by default and reveal predictions only
    // when they choose to (see revealPredictions below).
    const revealed = detail.review_exists || !hidePredictionsByDefault;
    setPredictionsRevealed(revealed);
    const initialNoCancer = detail.review_exists
      ? (detail.review_no_cancer ?? false)
      : revealed ? detail.registry_no_cancer : false;
    const initialCodes: CodeIn[] = detail.review_exists
      ? detail.review_codes.map((c) => ({ taxonomy_group: c.taxonomy_group, taxonomy_term: c.taxonomy_term }))
      : revealed ? detail.predicted_codes.map((c) => ({ taxonomy_group: c.cancer_type_name, taxonomy_term: c.predicted_term ?? '' })) : [];
    setNoCancer(initialNoCancer);
    setCodes(initialCodes);
    setBaselineNoCancer(initialNoCancer);
    setBaselineCodes(initialCodes);
  }, [detail, hidePredictionsByDefault]);

  // Reveals the model's predictions for the current case. Only pre-fills the
  // editable form from them if the reviewer hasn't already started their own
  // answer — revealing is informational and must never silently overwrite a
  // correction already in progress.
  const revealPredictions = useCallback(() => {
    if (!detail) return;
    setPredictionsRevealed(true);
    if (!noCancer && codes.length === 0) {
      const initialCodes: CodeIn[] = detail.predicted_codes.map((c) => ({
        taxonomy_group: c.cancer_type_name, taxonomy_term: c.predicted_term ?? '',
      }));
      setNoCancer(detail.registry_no_cancer);
      setCodes(initialCodes);
      setBaselineNoCancer(detail.registry_no_cancer);
      setBaselineCodes(initialCodes);
    }
  }, [detail, noCancer, codes]);

  const isUnmodifiedFromBaseline = useMemo(() => {
    if (noCancer !== baselineNoCancer) return false;
    if (codes.length !== baselineCodes.length) return false;
    const baselineSet = new Set(baselineCodes.map((c) => `${c.taxonomy_group}::${c.taxonomy_term}`));
    return codes.every((c) => baselineSet.has(`${c.taxonomy_group}::${c.taxonomy_term}`));
  }, [noCancer, codes, baselineNoCancer, baselineCodes]);

  const handleSave = useCallback(async () => {
    if (!detail) return;
    const token = await getAccessToken();
    if (!token) return;
    setSaving(true);
    setSaveError(null);
    setSaveSuccess(false);
    try {
      await saveCaseReview(token, detail.case_id, { no_cancer: noCancer, codes: noCancer ? [] : codes });
      setSaveSuccess(true);
      await loadCaseDetail(detail.case_id);
      await loadWorklist();
    } catch (e) {
      setSaveError(friendlyError(e, 'Save failed'));
    } finally {
      setSaving(false);
    }
  }, [detail, noCancer, codes, getAccessToken, loadCaseDetail, loadWorklist]);

  const handleReopen = useCallback(async () => {
    if (!detail) return;
    const token = await getAccessToken();
    if (!token) return;
    try {
      await reopenCaseReview(token, detail.case_id);
      await loadCaseDetail(detail.case_id);
      await loadWorklist();
    } catch (e) {
      setSaveError(friendlyError(e, 'Reopen failed'));
    }
  }, [detail, getAccessToken, loadCaseDetail, loadWorklist]);

  const handleImportList = useCallback(async () => {
    const listFile = listFileRef.current?.files?.[0];
    const manifestFile = manifestFileRef.current?.files?.[0];
    if (!listFile || !manifestFile) {
      setImportError('Select both the audit list .txt and its .manifest.json sidecar');
      return;
    }
    const token = await getAccessToken();
    if (!token) return;
    setImportingList(true);
    setImportSummary(null);
    setImportError(null);
    try {
      const summary = await importAuditList(token, listFile, manifestFile);
      let msg = `Imported ${summary.list_id} — ${summary.case_count} case(s).`;
      if (summary.replaced_list_id) msg += ` Replaced ${summary.replaced_list_id}.`;
      if (summary.not_found.length > 0) msg += ` ${summary.not_found.length} case(s) have no matching patient.`;
      setImportSummary(msg);
      if (listFileRef.current) listFileRef.current.value = '';
      if (manifestFileRef.current) manifestFileRef.current.value = '';
      await loadWorklist();
    } catch (e) {
      setImportError(friendlyError(e, 'Import failed'));
    } finally {
      setImportingList(false);
    }
  }, [getAccessToken, loadWorklist]);

  const handleExport = useCallback(async () => {
    if (!exportEmail.trim()) return;
    const token = await getAccessToken();
    if (!token) return;
    setExporting(true);
    setExportError(null);
    setExportResult(null);
    try {
      const summary = await createGoldExport(token, exportEmail.trim());
      setExportResult(summary);
      await loadWorklist();
    } catch (e) {
      setExportError(friendlyError(e, 'Export failed'));
    } finally {
      setExporting(false);
    }
  }, [exportEmail, getAccessToken, loadWorklist]);

  const handleDownload = useCallback(
    async (exportId: string) => {
      const token = await getAccessToken();
      if (!token) return;
      setDownloading(true);
      try {
        downloadBlob(await downloadGoldExport(token, exportId), `gold_${exportId}.csv`);
      } catch (e) {
        setExportError(friendlyError(e, 'Download failed'));
      } finally {
        setDownloading(false);
      }
    },
    [getAccessToken],
  );

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Audit Worklist</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Cases ML needs a human answer on. Record each case's complete cancer-code set, or mark it
          no reportable cancer — never both. Work from the top: the first cases matter most.
        </p>
        {worklist?.list_id && (
          <p className="text-xs text-gray-500 mt-2">
            List {worklist.list_id} · {worklist.case_count} case(s)
            {worklist.imported_at && ` · imported ${formatTs(worklist.imported_at)}`}
          </p>
        )}

        {isAdmin && (
          <div className="mt-3 border-t border-gray-100 pt-3">
            <button
              type="button"
              onClick={() => setShowAdminTools((v) => !v)}
              className="text-xs font-medium text-blue-600 hover:text-blue-800"
            >
              {showAdminTools ? '− Hide admin tools' : '+ Admin tools'}
            </button>
            {showAdminTools && (
              <div className="mt-3 space-y-4">
                <div>
                  <p className="text-xs font-medium text-gray-700 mb-1">Import audit list</p>
                  <div className="flex flex-wrap gap-2 items-center">
                    <label className="text-xs text-gray-600 flex items-center gap-1.5">
                      List (.txt)
                      <input ref={listFileRef} type="file" accept=".txt" disabled={importingList} className="text-xs w-36" />
                    </label>
                    <label className="text-xs text-gray-600 flex items-center gap-1.5">
                      Manifest (.manifest.json)
                      <input ref={manifestFileRef} type="file" accept=".json" disabled={importingList} className="text-xs w-44" />
                    </label>
                    <button
                      type="button"
                      onClick={handleImportList}
                      disabled={importingList}
                      className="px-2.5 py-1.5 text-xs font-medium bg-white text-gray-700 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                    >
                      {importingList ? 'Importing…' : 'Import'}
                    </button>
                  </div>
                  {importSummary && <p className="text-xs text-emerald-700 mt-1">{importSummary}</p>}
                  {importError && <p className="text-xs text-red-600 mt-1">{importError}</p>}
                </div>

                <div>
                  <p className="text-xs font-medium text-gray-700 mb-1">Export gold CSV for a reviewer</p>
                  <div className="flex flex-wrap gap-2 items-center">
                    <input
                      type="email"
                      placeholder="reviewer@ucdavis.edu"
                      value={exportEmail}
                      onChange={(e) => setExportEmail(e.target.value)}
                      className="w-56 px-2 py-1.5 border border-gray-300 rounded text-xs"
                    />
                    <button
                      type="button"
                      onClick={handleExport}
                      disabled={exporting || !exportEmail.trim()}
                      className="px-2.5 py-1.5 text-xs font-medium bg-white text-gray-700 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                    >
                      {exporting ? 'Exporting…' : 'Export'}
                    </button>
                  </div>
                  {exportResult && (
                    <p className="text-xs text-emerald-700 mt-1">
                      {exportResult.export_id} — {exportResult.case_count} case(s) locked.{' '}
                      <button
                        type="button"
                        onClick={() => handleDownload(exportResult.export_id)}
                        disabled={downloading}
                        className="underline hover:no-underline disabled:opacity-50"
                      >
                        {downloading ? 'Downloading…' : 'Download CSV'}
                      </button>
                    </p>
                  )}
                  {exportError && <p className="text-xs text-red-600 mt-1">{exportError}</p>}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {worklistError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-700">{worklistError}</p>
        </div>
      )}

      <div className="grid grid-cols-12 gap-4">
        {/* 720px ≈ the 44px header + 10 rows at 60px (+9px of divide-y
            borders) + the 45px pagination footer, plus a little slack —
            sized so a full page of 10 rows never needs its own scrollbar.
            The detail panel matches it so both panels have a constant,
            predictable height and scroll independently of each other. */}
        <div className="col-span-5 bg-white rounded-lg border border-gray-200 h-[720px] flex flex-col">
          <div className="px-4 py-3 border-b border-gray-200 shrink-0">
            <span className="text-sm font-medium">Worklist</span>
          </div>
          <div className="flex-1 overflow-y-auto">
            {loadingWorklist ? (
              <div className="p-6 text-sm text-gray-500">Loading...</div>
            ) : !worklist || worklist.cases.length === 0 ? (
              <div className="p-6 text-sm text-gray-500">
                {isAdmin ? 'No active audit list — import one above.' : 'No active audit list.'}
              </div>
            ) : (
              <ul className="divide-y divide-gray-100">
                {pagedCases.map((c) => {
                  const active = c.case_id === selectedCaseId;
                  return (
                    <li
                      key={c.case_id}
                      onClick={() => selectCase(c.case_id)}
                      className={`px-4 py-3 cursor-pointer hover:bg-gray-50 ${active ? 'bg-blue-50' : ''}`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-gray-900 truncate flex items-center gap-1.5">
                            {c.case_id}
                            {!c.patient_found && (
                              <span
                                title="No matching patient record"
                                className="shrink-0 inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-medium border bg-red-100 text-red-800 border-red-200"
                              >
                                No record
                              </span>
                            )}
                          </p>
                          <p className="text-xs text-gray-500 truncate">
                            #{c.position + 1}
                            {c.reviewed_by_email && ` · ${c.reviewed_by_email}`}
                          </p>
                        </div>
                        <StatusBadge status={c.review_status} />
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
          {!loadingWorklist && worklist && worklist.cases.length > 0 && totalWorklistPages > 1 && (
            <div className="shrink-0 flex items-center justify-between px-4 py-2.5 border-t border-gray-100">
              <button
                type="button"
                onClick={() => setWorklistPage((p) => Math.max(0, p - 1))}
                disabled={currentWorklistPage === 0}
                className="px-2.5 py-1 text-xs font-medium bg-white text-gray-700 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
              >
                Previous
              </button>
              <span className="text-xs text-gray-500">
                Page {currentWorklistPage + 1} of {totalWorklistPages}
              </span>
              <button
                type="button"
                onClick={() => setWorklistPage((p) => Math.min(totalWorklistPages - 1, p + 1))}
                disabled={currentWorklistPage === totalWorklistPages - 1}
                className="px-2.5 py-1 text-xs font-medium bg-white text-gray-700 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          )}
        </div>

        <div className="col-span-7 bg-white rounded-lg border border-gray-200 h-[720px] overflow-y-auto">
          {!selectedCaseId ? (
            <div className="p-6 text-sm text-gray-500">Select a case from the worklist.</div>
          ) : loadingDetail ? (
            <div className="p-6 text-sm text-gray-500">Loading...</div>
          ) : detailError ? (
            <div className="p-6 text-sm text-red-600">{detailError}</div>
          ) : detail ? (
            <div className="p-4 space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-gray-900">{detail.case_id}</h3>
                {!detail.patient_found ? (
                  <p className="text-xs text-red-600 mt-1">
                    No matching patient record — report text and predictions unavailable.
                  </p>
                ) : (
                  (detail.patient_species || detail.patient_breed || detail.patient_sex || detail.patient_age != null) && (
                    <p className="text-xs text-gray-500 mt-1">
                      {[
                        detail.patient_species,
                        detail.patient_breed,
                        detail.patient_sex,
                        detail.patient_age != null ? `${detail.patient_age} yr${detail.patient_age === 1 ? '' : 's'}` : null,
                      ].filter(Boolean).join(' · ')}
                    </p>
                  )
                )}
              </div>

              {detail.source_diagnosis && (
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">Clinical diagnosis</p>
                  <div className="text-sm text-gray-800 whitespace-pre-wrap max-h-48 overflow-auto border border-gray-100 rounded p-2 bg-gray-50">
                    {detail.source_diagnosis}
                  </div>
                </div>
              )}

              {detail.report_text && (
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">Pathology report</p>
                  <div className="text-sm text-gray-800 whitespace-pre-wrap max-h-48 overflow-auto border border-gray-100 rounded p-2 bg-gray-50">
                    {detail.report_text}
                  </div>
                </div>
              )}

              {detail.predicted_codes.length === 0 && !detail.registry_no_cancer ? null : !predictionsRevealed ? (
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">Current predictions</p>
                  <button
                    type="button"
                    onClick={revealPredictions}
                    className="text-xs font-medium text-blue-600 hover:text-blue-800 underline"
                  >
                    Hidden — click to reveal
                  </button>
                </div>
              ) : detail.predicted_codes.length > 0 ? (
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">Current predictions</p>
                  <ul className="text-sm text-gray-700 space-y-0.5">
                    {detail.predicted_codes.map((p, i) => (
                      <li key={i}>
                        {p.cancer_type_name}
                        {p.predicted_term && ` — ${p.predicted_term}`}
                        {p.confidence != null && (
                          <span className="text-gray-400"> ({(p.confidence * 100).toFixed(0)}%)</span>
                        )}
                        {p.code_source && (
                          <span className="text-gray-400"> · {CODE_SOURCE_LABELS[p.code_source] ?? p.code_source}</span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-xs text-gray-500">Registry: no reportable cancer.</p>
              )}

              <div className="border-t border-gray-100 pt-3">
                <p className="text-xs font-medium text-gray-500 mb-2">Gold review</p>

                {detail.review_locked && (
                  <div className="mb-3 flex items-center justify-between bg-blue-50 border border-blue-200 rounded p-2">
                    <p className="text-xs text-blue-800">
                      Locked — exported{detail.reviewed_by_email && ` by ${detail.reviewed_by_email}`}.
                    </p>
                    {isAdmin && (
                      <button
                        type="button"
                        onClick={handleReopen}
                        className="text-xs font-medium text-blue-700 hover:text-blue-900 underline"
                      >
                        Reopen
                      </button>
                    )}
                  </div>
                )}

                <fieldset disabled={!!detail.review_locked} className="space-y-3 disabled:opacity-60">
                  <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={noCancer} onChange={(e) => setNoCancer(e.target.checked)} />
                    No reportable cancer
                  </label>

                  {!noCancer && (
                    <div className="space-y-2">
                      <CodePicker
                        terms={terms}
                        codes={codes}
                        onAdd={(group, term) => {
                          setCodes((prev) =>
                            prev.some((c) => c.taxonomy_group === group && c.taxonomy_term === term)
                              ? prev
                              : [...prev, { taxonomy_group: group, taxonomy_term: term }],
                          );
                        }}
                      />
                      {codes.length > 0 && (
                        <ul className="space-y-1">
                          {codes.map((c, i) => (
                            <li
                              key={`${c.taxonomy_group}::${c.taxonomy_term}`}
                              className="flex items-center justify-between text-sm bg-gray-50 border border-gray-200 rounded px-2 py-1"
                            >
                              <span>
                                <span className="text-gray-500">{c.taxonomy_group}:</span> {c.taxonomy_term}
                              </span>
                              <button
                                type="button"
                                onClick={() => setCodes((prev) => prev.filter((_, idx) => idx !== i))}
                                className="text-gray-600 hover:text-red-600 text-xs font-medium"
                              >
                                Remove
                              </button>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={handleSave}
                    disabled={saving || (!noCancer && codes.length === 0)}
                    className="px-3 py-1.5 text-sm font-medium bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                  >
                    {saving ? 'Saving…' : isUnmodifiedFromBaseline ? 'Approve' : 'Save correction'}
                  </button>
                </fieldset>

                {saveSuccess && <p className="text-xs text-emerald-700 mt-2">Saved.</p>}
                {saveError && <p className="text-xs text-red-600 mt-2">{saveError}</p>}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
