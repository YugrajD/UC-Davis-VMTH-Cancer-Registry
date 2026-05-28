import { useState, useRef, useEffect, useCallback } from 'react';
import {
  uploadCSV, fetchMyJobs, fetchMyRoleRequests, submitRoleRequest,
  fetchMyExportRequests, submitExportRequest, downloadExportCsv,
  fetchFilterOptions, fetchIncidence, fetchCalEnviroScreen,
  type IngestionJob, type RoleRequest, type ExportRequest, type ExportFilters,
} from '../../api/client';
import { getHumanCancerRateMap } from '../../data/humanCancerRates';
import { useAuth } from '../../contexts/AuthContext';
import { LoginModal } from '../LoginModal/LoginModal';
import { STAGE_LABELS } from '../shared/pipelineStages';

interface CsvPreview {
  headers: string[];
  rows: string[][];
  totalRows: number;
}

function parseCsv(text: string): CsvPreview {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (lines.length === 0) return { headers: [], rows: [], totalRows: 0 };

  const parseLine = (line: string): string[] => {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (inQuotes) {
        if (ch === '"' && line[i + 1] === '"') {
          current += '"';
          i++;
        } else if (ch === '"') {
          inQuotes = false;
        } else {
          current += ch;
        }
      } else if (ch === '"') {
        inQuotes = true;
      } else if (ch === ',') {
        result.push(current.trim());
        current = '';
      } else {
        current += ch;
      }
    }
    result.push(current.trim());
    return result;
  };

  const headers = parseLine(lines[0]);
  const dataLines = lines.slice(1);
  const rows = dataLines.map(parseLine);
  return { headers, rows, totalRows: dataLines.length };
}

function PreviewModal({ file, onClose }: { file: File; onClose: () => void }) {
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        setPreview(parseCsv(e.target?.result as string));
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

const REQUIRED_COLUMNS = [
  'Date of Birth',
  'Sex',
  'Species',
  'Breed',
  'Zipcode Zipcode',
  'RfrrVtrn Zipcode Zipcode',
  'DtOfRq',
  'Text',
];

// Aliases that map alternate column names to their canonical required name
const COLUMN_ALIASES: Record<string, string> = {
  'text (pathology report)': 'Text',
};

/**
 * Validates that a CSV file contains all required columns.
 * Returns a list of missing column names, or empty array if valid.
 */
function validateCsvColumns(fileText: string): string[] {
  const firstLine = fileText.split(/\r?\n/).find(l => l.trim());
  if (!firstLine) return [...REQUIRED_COLUMNS];

  // Parse the header line using the same logic as parseCsv
  const headers: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < firstLine.length; i++) {
    const ch = firstLine[i];
    if (inQuotes) {
      if (ch === '"' && firstLine[i + 1] === '"') {
        current += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      headers.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  headers.push(current.trim());

  // Normalize headers: lowercase and apply aliases
  const normalizedHeaders = headers.map(h => {
    const lower = h.toLowerCase();
    const alias = COLUMN_ALIASES[lower];
    return alias ? alias.toLowerCase() : lower;
  });

  return REQUIRED_COLUMNS.filter(
    col => !normalizedHeaders.includes(col.toLowerCase())
  );
}

interface SignInPromptProps {
  onSignIn: () => void;
}

function SignInPrompt({ onSignIn }: SignInPromptProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
      <svg className="w-12 h-12 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
      <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">
        Sign in to upload data
      </h3>
      <p className="text-sm text-[var(--color-text-secondary)] mb-4">
        You need to be signed in to upload datasets for processing.
      </p>
      <button
        onClick={onSignIn}
        className="px-6 py-2.5 bg-[var(--color-teal)] text-white text-sm font-semibold rounded-md hover:bg-[var(--color-teal-dark)] transition-colors"
      >
        Sign In
      </button>
    </div>
  );
}

function RoleRequestCard({
  role,
  hasRole,
  pendingRequest,
  onSubmit,
  submitting,
}: {
  role: string;
  hasRole: boolean;
  pendingRequest: RoleRequest | undefined;
  onSubmit: (role: string, reason: string) => void;
  submitting: boolean;
}) {
  const [reason, setReason] = useState('');
  const label = role.charAt(0).toUpperCase() + role.slice(1);

  if (hasRole) {
    return (
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 border border-emerald-200">
          {label}
        </span>
      </div>
    );
  }

  if (pendingRequest) {
    return (
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 border border-yellow-200">
          {label}: pending
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <textarea
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        placeholder={`Why do you need ${role} access? (optional)`}
        rows={2}
        className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent resize-none"
      />
      <button
        onClick={() => onSubmit(role, reason)}
        disabled={submitting}
        className="px-4 py-2 text-sm font-medium bg-[var(--color-teal)] text-white rounded-md hover:bg-[var(--color-teal-dark)] disabled:opacity-50 transition-colors"
      >
        {submitting ? 'Requesting...' : `Request ${label} Role`}
      </button>
    </div>
  );
}

type SubTab = 'dataset' | 'export';

export function DataUpload() {
  const { user, isUploader, isReviewer, isAdmin, getAccessToken } = useAuth();
  const [subTab, setSubTab] = useState<SubTab>('dataset');
  const [file, setFile] = useState<File | null>(null);
  const [clinicName, setClinicName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [myJobs, setMyJobs] = useState<IngestionJob[]>([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [showLogin, setShowLogin] = useState(false);

  // Role requests
  const [roleRequests, setRoleRequests] = useState<RoleRequest[]>([]);
  const [roleSubmitting, setRoleSubmitting] = useState(false);
  const [roleError, setRoleError] = useState<string | null>(null);

  // Export requests
  const [exportRequests, setExportRequests] = useState<ExportRequest[]>([]);
  const [exportReason, setExportReason] = useState('');
  const [exportSubmitting, setExportSubmitting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportDownloading, setExportDownloading] = useState(false);
  const [combinedDownloading, setCombinedDownloading] = useState(false);
  const [exportCancerType, setExportCancerType] = useState('');
  const [exportCancerTypes, setExportCancerTypes] = useState<{ id: number; name: string }[]>([]);
  const [exportCounty, setExportCounty] = useState('');
  const [exportCounties, setExportCounties] = useState<{ id: number; name: string }[]>([]);
  const [exportZipCode, setExportZipCode] = useState('');
  const [exportSex, setExportSex] = useState('');
  const [exportBreed, setExportBreed] = useState('');
  const [exportBreeds, setExportBreeds] = useState<{ id: number; name: string }[]>([]);
  const [exportYearStart, setExportYearStart] = useState('');
  const [exportYearEnd, setExportYearEnd] = useState('');
  const [exportYearRange, setExportYearRange] = useState<number[]>([]);

  const fileRef = useRef<HTMLInputElement>(null);

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

  const loadRoleRequests = useCallback(async () => {
    const token = await getAccessToken();
    if (!token) return;
    try {
      const reqs = await fetchMyRoleRequests(token);
      setRoleRequests(reqs);
    } catch {
      // silently fail
    }
  }, [getAccessToken]);

  const loadExportRequests = useCallback(async () => {
    const token = await getAccessToken();
    if (!token) return;
    try {
      const reqs = await fetchMyExportRequests(token);
      setExportRequests(reqs);
    } catch {
      // silently fail
    }
  }, [getAccessToken]);

  useEffect(() => {
    if (user) {
      loadMyJobs(); // eslint-disable-line react-hooks/set-state-in-effect
      loadRoleRequests();
      loadExportRequests();
    }
  }, [user, loadMyJobs, loadRoleRequests, loadExportRequests]);

  useEffect(() => {
    fetchFilterOptions()
      .then((opts) => {
        setExportCancerTypes(opts.cancer_types);
        setExportCounties(opts.counties);
        setExportBreeds(opts.breeds);
        setExportYearRange(opts.year_range);
      })
      .catch(() => {});
  }, []);

  // Auto-poll every 10s while any job is processing
  useEffect(() => {
    const hasProcessing = myJobs.some(j => j.status === 'processing');
    if (!hasProcessing) return;
    const interval = setInterval(loadMyJobs, 10000);
    return () => clearInterval(interval);
  }, [myJobs, loadMyJobs]);

  const validateFile = useCallback((f: File): Promise<string[]> => {
    return new Promise((resolve) => {
      // Skip validation for non-CSV files (xlsx handled by backend)
      if (!f.name.toLowerCase().endsWith('.csv')) {
        resolve([]);
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        resolve(validateCsvColumns(text));
      };
      reader.onerror = () => resolve([]); // Don't block upload on read error
      // Only read the first 4KB to get headers
      reader.readAsText(f.slice(0, 4096));
    });
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null;
    setFile(selected);
    setError(null);
    setSubmitted(false);

    if (selected) {
      const missing = await validateFile(selected);
      if (missing.length > 0) {
        setError(`Missing required columns: ${missing.join(', ')}`);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a dataset file.');
      return;
    }
    if (!clinicName.trim()) {
      setError('Please enter a clinic name.');
      return;
    }

    // Re-validate before upload
    const missing = await validateFile(file);
    if (missing.length > 0) {
      setError(`Missing required columns: ${missing.join(', ')}`);
      return;
    }

    setLoading(true);
    setError(null);
    setSubmitted(false);

    try {
      const token = await getAccessToken();
      if (!token) {
        setError('Not signed in');
        return;
      }
      await uploadCSV(file, token, clinicName.trim());
      setSubmitted(true);
      await loadMyJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setClinicName('');
    setSubmitted(false);
    setError(null);
    if (fileRef.current) fileRef.current.value = '';
  };

  const handleRoleRequest = async (role: string, reason: string) => {
    setRoleError(null);
    setRoleSubmitting(true);
    try {
      const token = await getAccessToken();
      if (!token) return;
      await submitRoleRequest(token, role, reason || undefined);
      await loadRoleRequests();
    } catch (err) {
      setRoleError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setRoleSubmitting(false);
    }
  };

  const handleExportRequest = async () => {
    setExportError(null);
    setExportSubmitting(true);
    try {
      const token = await getAccessToken();
      if (!token) return;
      await submitExportRequest(token, exportReason || undefined);
      setExportReason('');
      await loadExportRequests();
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setExportSubmitting(false);
    }
  };

  const handleExportDownload = async () => {
    setExportError(null);
    setExportDownloading(true);
    try {
      const token = await getAccessToken();
      if (!token) return;
      const filters: ExportFilters = {
        cancerType: exportCancerType || undefined,
        county: exportCounty || undefined,
        zipCode: exportZipCode || undefined,
        sex: exportSex || undefined,
        breed: exportBreed || undefined,
        yearStart: exportYearStart ? Number(exportYearStart) : undefined,
        yearEnd: exportYearEnd ? Number(exportYearEnd) : undefined,
      };
      const blob = await downloadExportCsv(token, filters);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'vmth_patient_export.csv';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      // Reload requests so the consumed approval is reflected in the UI
      await loadExportRequests();
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Download failed');
    } finally {
      setExportDownloading(false);
    }
  };

  const handleCombinedDownload = async () => {
    setCombinedDownloading(true);
    try {
      // Pass the same filters the user set for the patient export.
      // zipCode and breed are not supported by the county-level incidence
      // endpoint so they are intentionally omitted here.
      const incidenceFilters = {
        ...(exportCancerType && { cancerTypes: [exportCancerType] }),
        ...(exportCounty && { counties: [exportCounty] }),
        ...(exportSex && { sex: exportSex }),
        ...(exportYearStart && { yearStart: Number(exportYearStart) }),
        ...(exportYearEnd && { yearEnd: Number(exportYearEnd) }),
      };
      const [incidenceRes, cesData] = await Promise.all([
        fetchIncidence(incidenceFilters),
        fetchCalEnviroScreen(),
      ]);

      const hrMap = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');

      // Aggregate dog cancer cases by county name
      const dogCasesMap = new Map<string, number>();
      for (const rec of incidenceRes.data) {
        if (rec.county) {
          dogCasesMap.set(rec.county, (dogCasesMap.get(rec.county) ?? 0) + rec.count);
        }
      }

      const headers = [
        'county',
        'dog_cancer_cases',
        'human_cancer_rate_per_100k',
        'human_cancer_cases_per_year',
        'ces_score', 'pollution_burden', 'ozone', 'pm25', 'diesel_pm',
        'pesticides', 'toxic_releases', 'traffic', 'drinking_water', 'lead',
        'cleanup_sites', 'groundwater_threats', 'hazardous_waste', 'solid_waste',
        'impaired_water', 'pop_characteristics', 'asthma', 'low_birth_weight',
        'cardiovascular', 'poverty', 'unemployment', 'housing_burden',
        'education', 'linguistic_isolation',
      ];

      const toCell = (v: number | null | undefined) => v != null ? String(v) : '';
      const quoteCell = (s: string) => `"${s.replace(/"/g, '""')}"`;

      const rows = cesData.map(ces => {
        const hr = hrMap.get(ces.county_name.toLowerCase());
        const dogCases = dogCasesMap.get(ces.county_name) ?? 0;
        return [
          ces.county_name, String(dogCases),
          toCell(hr?.rate), toCell(hr?.cases),
          toCell(ces.ces_score), toCell(ces.pollution_burden), toCell(ces.ozone),
          toCell(ces.pm25), toCell(ces.diesel_pm), toCell(ces.pesticides),
          toCell(ces.toxic_releases), toCell(ces.traffic), toCell(ces.drinking_water),
          toCell(ces.lead), toCell(ces.cleanup_sites), toCell(ces.groundwater_threats),
          toCell(ces.hazardous_waste), toCell(ces.solid_waste), toCell(ces.impaired_water),
          toCell(ces.pop_characteristics), toCell(ces.asthma), toCell(ces.low_birth_weight),
          toCell(ces.cardiovascular), toCell(ces.poverty), toCell(ces.unemployment),
          toCell(ces.housing_burden), toCell(ces.education), toCell(ces.linguistic_isolation),
        ].map(quoteCell).join(',');
      });

      const csv = [headers.join(','), ...rows].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'vmth_combined_data.csv';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Download failed');
    } finally {
      setCombinedDownloading(false);
    }
  };

  const pendingUploader = roleRequests.find(r => r.requested_role === 'uploader' && r.status === 'pending');
  const pendingReviewer = roleRequests.find(r => r.requested_role === 'reviewer' && r.status === 'pending');
  const pendingExport = exportRequests.find(r => r.status === 'pending');
  const approvedExport = exportRequests.find(r => r.status === 'approved');

  // Not signed in — show sign-in prompt
  if (!user) {
    return (
      <div className="space-y-6">
        <SignInPrompt onSignIn={() => setShowLogin(true)} />
        {showLogin && (
          <LoginModal onClose={() => setShowLogin(false)} />
        )}
      </div>
    );
  }

  const SUB_TABS: { id: SubTab; label: string }[] = [
    { id: 'dataset', label: 'Dataset' },
    { id: 'export', label: 'Data Export' },
  ];

  return (
    <div className="space-y-6">
      {/* Sub-tab bar */}
      <div className="flex gap-1 border-b border-gray-200">
        {SUB_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setSubTab(tab.id)}
            className={`px-5 py-2.5 text-sm font-medium transition-all duration-200 border-b-2 -mb-[1px] ${
              subTab === tab.id
                ? 'text-[var(--color-teal-dark)] border-[var(--color-teal)]'
                : 'text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {subTab === 'dataset' && (
        <>
          {/* File Input */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider mb-1">
              Dataset
            </h3>
            <p className="text-xs text-[var(--color-text-secondary)] mb-3">
              Upload a CSV (.csv) or Excel (.xlsx) file containing patient visit data. Required columns:
            </p>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {REQUIRED_COLUMNS.map(col => (
                <code key={col} className="bg-gray-100 text-gray-700 px-1.5 py-0.5 rounded text-xs">
                  {col}
                </code>
              ))}
            </div>
            <a
              href="/upload_template.csv"
              download="upload_template.csv"
              className="inline-flex items-center gap-1 text-xs text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium transition-colors mb-4"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download template CSV
            </a>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Clinic name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={clinicName}
                onChange={(e) => setClinicName(e.target.value)}
                placeholder="e.g. UC Davis VMTH"
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
              />
            </div>
            <label className="block">
              <span className="sr-only">Choose dataset file</span>
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.xlsx"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-[var(--color-teal)] file:text-white
                  hover:file:bg-[var(--color-teal-dark)] file:cursor-pointer"
              />
            </label>
            {file && (
              <>
                <p className="mt-2 text-xs text-green-700">{file.name} selected</p>
                <FilePreview file={file} />
              </>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-4">
            <button
              onClick={handleUpload}
              disabled={loading || !file || !clinicName.trim()}
              className="px-6 py-2.5 bg-[var(--color-teal)] text-white text-sm font-semibold rounded-md
                hover:bg-[var(--color-teal-dark)] disabled:opacity-50 disabled:cursor-not-allowed
                transition-colors"
            >
              {loading ? 'Submitting...' : 'Submit for Review'}
            </button>
            {(file || submitted) && (
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
              <p className="text-sm text-blue-800">Uploading file for review...</p>
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

          {/* My Uploads */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
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
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">File</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Submitted</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {myJobs.map((job) => (
                      <tr key={job.id} className="hover:bg-gray-50">
                        <td className="px-4 py-2.5 text-gray-600">#{job.id}</td>
                        <td className="px-4 py-2.5 font-mono text-xs">{job.dataset_a_filename}</td>
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
                          {job.status === 'completed' && job.result_summary && (
                            <div className="mt-1 space-y-0.5">
                              <p className="text-xs text-gray-500">
                                {job.result_summary.patients} records · {job.result_summary.diagnoses} diagnoses
                              </p>
                              {job.result_summary.avg_confidence !== null && (
                                <p className="text-xs font-medium text-green-700">
                                  {job.result_summary.avg_confidence}% avg confidence
                                </p>
                              )}
                              {job.result_summary.top_cancer_types?.[0] && (
                                <p className="text-xs text-gray-500 truncate max-w-[180px]" title={job.result_summary.top_cancer_types[0].name}>
                                  Top: {job.result_summary.top_cancer_types[0].name}
                                </p>
                              )}
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
          </div>

          {/* Role Requests */}
          {!isAdmin && (
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider mb-1">
                Request Roles
              </h3>
              <p className="text-xs text-[var(--color-text-secondary)] mb-4">
                Request additional permissions. An admin will review your request.
              </p>

              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Uploader</p>
                  <p className="text-xs text-gray-500 mb-2">Bypasses the 3-uploads-per-day rate limit.</p>
                  <RoleRequestCard
                    role="uploader"
                    hasRole={isUploader}
                    pendingRequest={pendingUploader}
                    onSubmit={handleRoleRequest}
                    submitting={roleSubmitting}
                  />
                </div>
                <div className="border-t border-gray-200 pt-4">
                  <p className="text-sm font-medium text-gray-700 mb-2">Reviewer</p>
                  <p className="text-xs text-gray-500 mb-2">Access to the Review Queue and Diagnosis Review tabs.</p>
                  <RoleRequestCard
                    role="reviewer"
                    hasRole={isReviewer}
                    pendingRequest={pendingReviewer}
                    onSubmit={handleRoleRequest}
                    submitting={roleSubmitting}
                  />
                </div>
              </div>

              {roleError && (
                <div className="mt-4 bg-red-50 border border-red-200 rounded-md p-3">
                  <p className="text-sm text-red-700">{roleError}</p>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {subTab === 'export' && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <p className="text-xs text-[var(--color-text-secondary)] mb-4">
            Download patient-level disease and demographic data as CSV. Each approved request grants a single download.
          </p>

          {isAdmin || approvedExport ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label htmlFor="export-cancer-type" className="block text-xs font-medium text-gray-700 mb-1">
                    Cancer Type
                  </label>
                  <select
                    id="export-cancer-type"
                    value={exportCancerType}
                    onChange={(e) => setExportCancerType(e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                  >
                    <option value="">All Cancer Types</option>
                    {exportCancerTypes.map((ct) => (
                      <option key={ct.id} value={ct.name}>{ct.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label htmlFor="export-county" className="block text-xs font-medium text-gray-700 mb-1">
                    County
                  </label>
                  <select
                    id="export-county"
                    value={exportCounty}
                    onChange={(e) => setExportCounty(e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                  >
                    <option value="">All Counties</option>
                    {exportCounties.map((c) => (
                      <option key={c.id} value={c.name}>{c.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label htmlFor="export-zip-code" className="block text-xs font-medium text-gray-700 mb-1">
                    Zip Code
                  </label>
                  <input
                    id="export-zip-code"
                    type="text"
                    value={exportZipCode}
                    onChange={(e) => setExportZipCode(e.target.value)}
                    placeholder="e.g. 95616"
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                  />
                </div>
                <div>
                  <label htmlFor="export-sex" className="block text-xs font-medium text-gray-700 mb-1">
                    Sex
                  </label>
                  <select
                    id="export-sex"
                    value={exportSex}
                    onChange={(e) => setExportSex(e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                  >
                    <option value="">All</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Neutered Male">Neutered Male</option>
                    <option value="Spayed Female">Spayed Female</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="export-breed" className="block text-xs font-medium text-gray-700 mb-1">
                    Breed
                  </label>
                  <select
                    id="export-breed"
                    value={exportBreed}
                    onChange={(e) => setExportBreed(e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                  >
                    <option value="">All Breeds</option>
                    {exportBreeds.map((b) => (
                      <option key={b.id} value={b.name}>{b.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Year Range
                  </label>
                  <div className="flex gap-2">
                    <input
                      id="export-year-start"
                      type="number"
                      value={exportYearStart}
                      onChange={(e) => setExportYearStart(e.target.value)}
                      placeholder={exportYearRange[0] ? String(exportYearRange[0]) : 'Start'}
                      min={exportYearRange[0] || undefined}
                      max={exportYearRange[1] || undefined}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                    />
                    <input
                      id="export-year-end"
                      type="number"
                      value={exportYearEnd}
                      onChange={(e) => setExportYearEnd(e.target.value)}
                      placeholder={exportYearRange[1] ? String(exportYearRange[1]) : 'End'}
                      min={exportYearRange[0] || undefined}
                      max={exportYearRange[1] || undefined}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                    />
                  </div>
                </div>
              </div>
              <button
                onClick={handleExportDownload}
                disabled={exportDownloading}
                className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium bg-[var(--color-teal)] text-white rounded-md hover:bg-[var(--color-teal-dark)] disabled:opacity-50 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                {exportDownloading ? 'Downloading...' : 'Download Export CSV'}
              </button>

              <div className="border-t border-gray-200 pt-3">
                <p className="text-xs text-[var(--color-text-secondary)] mb-2">
                  County-level combined dataset: dog cancer cases, human cancer rates (NCI/CDC 2017–2021), and CalEnviroScreen environmental indicators.
                </p>
                <button
                  onClick={handleCombinedDownload}
                  disabled={combinedDownloading}
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium bg-white text-[var(--color-teal)] border border-[var(--color-teal)] rounded-md hover:bg-teal-50 disabled:opacity-50 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  {combinedDownloading ? 'Preparing...' : 'Download Combined Data CSV'}
                </button>
              </div>
            </div>
          ) : pendingExport ? (
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 border border-yellow-200">
                Pending approval
              </span>
              <span className="text-xs text-gray-500">
                Submitted {new Date(pendingExport.created_at).toLocaleString()}
              </span>
            </div>
          ) : (
            <div className="space-y-2">
              <textarea
                value={exportReason}
                onChange={(e) => setExportReason(e.target.value)}
                placeholder="Why do you need data export access? (optional)"
                rows={2}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent resize-none"
              />
              <button
                onClick={handleExportRequest}
                disabled={exportSubmitting}
                className="px-4 py-2 text-sm font-medium bg-[var(--color-teal)] text-white rounded-md hover:bg-[var(--color-teal-dark)] disabled:opacity-50 transition-colors"
              >
                {exportSubmitting ? 'Requesting...' : 'Request Data Export'}
              </button>
            </div>
          )}

          {exportError && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-md p-3">
              <p className="text-sm text-red-700">{exportError}</p>
            </div>
          )}

          {/* Past export requests */}
          {exportRequests.length > 0 && (
            <div className="mt-4 border-t border-gray-200 pt-4">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Export Request History</p>
              <div className="space-y-2">
                {exportRequests.map((req) => (
                  <div key={req.id} className="flex items-center gap-2 text-xs">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full font-medium border ${
                      req.status === 'approved' ? 'bg-emerald-100 text-emerald-800 border-emerald-200'
                      : req.status === 'downloaded' ? 'bg-blue-100 text-blue-800 border-blue-200'
                      : req.status === 'denied' ? 'bg-red-100 text-red-800 border-red-200'
                      : 'bg-yellow-100 text-yellow-800 border-yellow-200'
                    }`}>
                      {req.status}
                    </span>
                    <span className="text-gray-500">
                      {new Date(req.created_at).toLocaleString()}
                    </span>
                    {req.reason && (
                      <span className="text-gray-400 truncate max-w-[200px]" title={req.reason}>
                        {req.reason}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

