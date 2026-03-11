import { useState, useRef, useCallback } from 'react';
import { uploadCSV, type IngestionResponse, type IngestionRowResult } from '../../api/client';

const PREVIEW_ROWS = 5;

interface CsvPreview {
  headers: string[];
  rows: string[][];
  totalRows: number;
}

function parseCsvPreview(text: string): CsvPreview {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (lines.length === 0) return { headers: [], rows: [], totalRows: 0 };

  // Simple CSV parse (handles quoted fields with commas)
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
  const rows = dataLines.slice(0, PREVIEW_ROWS).map(parseLine);
  return { headers, rows, totalRows: dataLines.length };
}

function FilePreview({ file }: { file: File }) {
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadPreview = useCallback(() => {
    if (preview) {
      setOpen(o => !o);
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        setPreview(parseCsvPreview(text));
        setOpen(true);
      } catch {
        setError('Could not parse file');
      }
    };
    reader.onerror = () => setError('Could not read file');
    reader.readAsText(file);
  }, [file, preview]);

  return (
    <div className="mt-3">
      <button
        onClick={loadPreview}
        className="text-xs text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium transition-colors"
      >
        {open ? 'Hide preview' : 'Preview file'}
      </button>
      {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
      {open && preview && (
        <div className="mt-2 border border-gray-200 rounded-lg overflow-hidden">
          <div className="overflow-x-auto max-h-64">
            <table className="w-full text-xs">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  {preview.headers.map((h, i) => (
                    <th key={i} className="px-3 py-1.5 text-left font-medium text-gray-500 uppercase whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {preview.rows.map((row, ri) => (
                  <tr key={ri} className="hover:bg-gray-50">
                    {row.map((cell, ci) => (
                      <td key={ci} className="px-3 py-1.5 text-gray-700 whitespace-nowrap max-w-[200px] truncate">
                        {cell || <span className="text-gray-300">—</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-3 py-1.5 bg-gray-50 border-t border-gray-200 text-[10px] text-gray-400">
            Showing {Math.min(PREVIEW_ROWS, preview.rows.length)} of {preview.totalRows} rows
          </div>
        </div>
      )}
    </div>
  );
}

function StatusIcon({ status }: { status: IngestionRowResult['status'] }) {
  if (status === 'inserted') return <span className="text-green-600 font-bold">+</span>;
  if (status === 'skipped') return <span className="text-yellow-600 font-bold">~</span>;
  return <span className="text-red-600 font-bold">!</span>;
}

export function DataUpload() {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IngestionResponse | null>(null);

  const refA = useRef<HTMLInputElement>(null);
  const refB = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    if (!fileA && !fileB) {
      setError('Please select at least Dataset A (PetBERT Predictions).');
      return;
    }
    if (!fileA && fileB) {
      setError('Dataset A (PetBERT Predictions) is required to process data.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await uploadCSV(fileA ?? undefined, fileB ?? undefined);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFileA(null);
    setFileB(null);
    setResult(null);
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
            Dataset A — PetBERT Predictions
          </h3>
          <p className="text-xs text-[var(--color-text-secondary)] mb-4">
            CSV with columns:{' '}
            <code className="bg-gray-100 px-1 rounded">anon_id</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">original_text</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">predicted_term</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">predicted_group</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">predicted_code</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">confidence</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">method</code>
          </p>
          <label className="block">
            <span className="sr-only">Choose Dataset A file</span>
            <input
              ref={refA}
              type="file"
              accept=".csv"
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
            <code className="bg-gray-100 px-1 rounded">case_id</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">DtOfRq</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">Sex</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">Species</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">Breed</code>
          </p>
          <label className="block">
            <span className="sr-only">Choose Dataset B file</span>
            <input
              ref={refB}
              type="file"
              accept=".csv"
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
          disabled={loading || (!fileA && !fileB)}
          className="px-6 py-2.5 bg-[var(--color-teal)] text-white text-sm font-semibold rounded-md
            hover:bg-[var(--color-teal-dark)] disabled:opacity-50 disabled:cursor-not-allowed
            transition-colors"
        >
          {loading ? 'Processing...' : 'Upload & Process'}
        </button>
        {(fileA || fileB || result) && (
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
          <p className="text-sm text-blue-800">
            Parsing CSVs, classifying pathology reports, and storing results...
          </p>
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

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Summary stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg border border-gray-200 p-4 text-center">
              <p className="text-2xl font-bold text-[var(--color-text-primary)]">{result.total_rows}</p>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">Total Rows</p>
            </div>
            <div className="bg-white rounded-lg border border-green-200 p-4 text-center">
              <p className="text-2xl font-bold text-green-700">{result.inserted}</p>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">Inserted</p>
            </div>
            <div className="bg-white rounded-lg border border-yellow-200 p-4 text-center">
              <p className="text-2xl font-bold text-yellow-700">{result.skipped}</p>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">Skipped</p>
            </div>
            <div className="bg-white rounded-lg border border-red-200 p-4 text-center">
              <p className="text-2xl font-bold text-red-700">{result.errors}</p>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">Errors</p>
            </div>
          </div>

          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
              <p className="text-sm font-semibold text-orange-800 mb-2">
                Warnings ({result.warnings.length})
              </p>
              <ul className="space-y-1">
                {result.warnings.map((w, i) => (
                  <li key={i} className="text-xs text-orange-700">
                    {w}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Row-level results table */}
          {result.row_results.length > 0 && (
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-200">
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
                  Row Details
                </h3>
              </div>
              <div className="overflow-x-auto max-h-96 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">#</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Anon ID</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Cancer Type</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Message</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {result.row_results.map((row) => (
                      <tr key={row.row_number} className="hover:bg-gray-50">
                        <td className="px-4 py-2 text-gray-500">{row.row_number}</td>
                        <td className="px-4 py-2 font-mono text-xs">{row.anon_id}</td>
                        <td className="px-4 py-2">
                          <span className="inline-flex items-center gap-1">
                            <StatusIcon status={row.status} />
                            <span className={
                              row.status === 'inserted' ? 'text-green-700' :
                              row.status === 'skipped' ? 'text-yellow-700' :
                              'text-red-700'
                            }>
                              {row.status}
                            </span>
                          </span>
                        </td>
                        <td className="px-4 py-2">{row.cancer_type ?? '—'}</td>
                        <td className="px-4 py-2">
                          {row.confidence != null ? (
                            <span className={row.confidence < 0.3 ? 'text-orange-600 font-medium' : ''}>
                              {(row.confidence * 100).toFixed(1)}%
                            </span>
                          ) : '—'}
                        </td>
                        <td className="px-4 py-2 text-xs text-gray-500">{row.message ?? ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
