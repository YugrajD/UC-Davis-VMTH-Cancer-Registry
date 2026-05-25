import { describe, it, expect } from 'vitest';
import type { IncidenceRecord } from '../api/client';

// ---------------------------------------------------------------------------
// Helpers mirrored from handleCombinedDownload in DataUpload.tsx
// These are pure functions inlined in the handler — tested here as logic units.
// ---------------------------------------------------------------------------

function buildDogCasesMap(records: IncidenceRecord[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const rec of records) {
    if (rec.county) {
      m.set(rec.county, (m.get(rec.county) ?? 0) + rec.count);
    }
  }
  return m;
}

const toCell = (v: number | null | undefined): string => (v != null ? String(v) : '');
const quoteCell = (s: string): string => `"${s.replace(/"/g, '""')}"`;

// ---------------------------------------------------------------------------
// buildDogCasesMap — county aggregation
// ---------------------------------------------------------------------------

describe('buildDogCasesMap — aggregation', () => {
  it('returns an empty map for no records', () => {
    expect(buildDogCasesMap([]).size).toBe(0);
  });

  it('aggregates a single record', () => {
    const records: IncidenceRecord[] = [
      { cancer_type: 'Lymphoma', county: 'Sacramento', count: 5 },
    ];
    const m = buildDogCasesMap(records);
    expect(m.get('Sacramento')).toBe(5);
  });

  it('sums multiple records for the same county', () => {
    const records: IncidenceRecord[] = [
      { cancer_type: 'Lymphoma', county: 'Sacramento', count: 5 },
      { cancer_type: 'Osteosarcoma', county: 'Sacramento', count: 3 },
      { cancer_type: 'Mast Cell', county: 'Sacramento', count: 2 },
    ];
    const m = buildDogCasesMap(records);
    expect(m.get('Sacramento')).toBe(10);
  });

  it('keeps different counties separate', () => {
    const records: IncidenceRecord[] = [
      { cancer_type: 'Lymphoma', county: 'Sacramento', count: 5 },
      { cancer_type: 'Lymphoma', county: 'Yolo', count: 2 },
    ];
    const m = buildDogCasesMap(records);
    expect(m.get('Sacramento')).toBe(5);
    expect(m.get('Yolo')).toBe(2);
  });

  it('skips records with no county', () => {
    const records: IncidenceRecord[] = [
      { cancer_type: 'Lymphoma', count: 10 },
      { cancer_type: 'Osteosarcoma', county: 'Yolo', count: 3 },
    ];
    const m = buildDogCasesMap(records);
    expect(m.size).toBe(1);
    expect(m.get('Yolo')).toBe(3);
  });

  it('is case-sensitive — Sacramento and sacramento are separate keys', () => {
    const records: IncidenceRecord[] = [
      { cancer_type: 'Lymphoma', county: 'Sacramento', count: 5 },
      { cancer_type: 'Lymphoma', county: 'sacramento', count: 2 },
    ];
    const m = buildDogCasesMap(records);
    expect(m.size).toBe(2);
  });

  it('is deterministic', () => {
    const records: IncidenceRecord[] = [
      { cancer_type: 'Lymphoma', county: 'Yolo', count: 4 },
      { cancer_type: 'Osteosarcoma', county: 'Yolo', count: 1 },
    ];
    const a = buildDogCasesMap(records);
    const b = buildDogCasesMap(records);
    expect(a.get('Yolo')).toBe(b.get('Yolo'));
  });
});

// ---------------------------------------------------------------------------
// toCell — numeric serialisation
// ---------------------------------------------------------------------------

describe('toCell', () => {
  it('converts a number to its string representation', () => {
    expect(toCell(42)).toBe('42');
  });

  it('handles 0 as "0" (not empty)', () => {
    expect(toCell(0)).toBe('0');
  });

  it('returns empty string for null', () => {
    expect(toCell(null)).toBe('');
  });

  it('returns empty string for undefined', () => {
    expect(toCell(undefined)).toBe('');
  });

  it('preserves decimal precision', () => {
    expect(toCell(3.14)).toBe('3.14');
  });
});

// ---------------------------------------------------------------------------
// quoteCell — CSV cell escaping
// ---------------------------------------------------------------------------

describe('quoteCell', () => {
  it('wraps a plain string in double quotes', () => {
    expect(quoteCell('hello')).toBe('"hello"');
  });

  it('escapes internal double quotes by doubling them', () => {
    expect(quoteCell('say "hi"')).toBe('"say ""hi"""');
  });

  it('handles empty string', () => {
    expect(quoteCell('')).toBe('""');
  });

  it('handles string that is only quotes', () => {
    expect(quoteCell('"')).toBe('""""');
  });

  it('preserves commas inside the cell', () => {
    expect(quoteCell('a,b,c')).toBe('"a,b,c"');
  });

  it('preserves newlines inside the cell', () => {
    expect(quoteCell('line1\nline2')).toBe('"line1\nline2"');
  });
});

// ---------------------------------------------------------------------------
// CSV row structure (integration of the two helpers)
// ---------------------------------------------------------------------------

describe('CSV row assembly', () => {
  it('a row with all values produces the expected quoted CSV line', () => {
    const cells = ['Sacramento', '10', '371.6', '6999'];
    const line = cells.map(quoteCell).join(',');
    expect(line).toBe('"Sacramento","10","371.6","6999"');
  });

  it('a row with null values produces empty quoted cells', () => {
    const cells = ['Alpine', '0', toCell(null), toCell(null)];
    const line = cells.map(quoteCell).join(',');
    expect(line).toBe('"Alpine","0","",""');
  });

  it('county name containing a comma is safely quoted', () => {
    const cells = ['County, CA', '5'];
    const line = cells.map(quoteCell).join(',');
    expect(line).toBe('"County, CA","5"');
  });
});
