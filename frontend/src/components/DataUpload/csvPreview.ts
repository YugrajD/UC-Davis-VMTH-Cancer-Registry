export interface CsvPreview {
  headers: string[];
  rows: string[][];
  totalRows: number;
}

function parseLine(line: string): string[] {
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
}

export function parseCsvPreview(text: string): CsvPreview {
  const lines = text.split(/\r?\n/).filter(line => line.trim());
  if (lines.length === 0) return { headers: [], rows: [], totalRows: 0 };

  const headers = parseLine(lines[0]);
  const dataLines = lines.slice(1);
  const rows = dataLines.map(parseLine);
  return { headers, rows, totalRows: dataLines.length };
}
