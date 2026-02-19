import { useState, useRef } from 'react';
import { uploadCSV, type IngestionResponse, type IngestionRowResult } from '../../api/client';

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
      setError('Please select at least Dataset A (Clinical Notes).');
      return;
    }
    if (!fileA && fileB) {
      setError('Dataset A (Clinical Notes) is required to process data.');
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
            Dataset A — Clinical Notes
          </h3>
          <p className="text-xs text-[var(--color-text-secondary)] mb-4">
            CSV with columns: <code className="bg-gray-100 px-1 rounded">anon_id</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">clinical_notes</code>
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
            <p className="mt-2 text-xs text-green-700">{fileA.name} selected</p>
          )}
        </div>

        {/* Dataset B */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider mb-1">
            Dataset B — Demographics
          </h3>
          <p className="text-xs text-[var(--color-text-secondary)] mb-4">
            CSV with columns: <code className="bg-gray-100 px-1 rounded">anon_id</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">zip_code</code>,{' '}
            <code className="bg-gray-100 px-1 rounded">sex</code>
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
            <p className="mt-2 text-xs text-green-700">{fileB.name} selected</p>
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
