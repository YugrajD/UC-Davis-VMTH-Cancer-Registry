// Client-side search over the Vet-ICD-O-canine-1 taxonomy for the review
// screen's Code/Term picker (see CodePicker.tsx). Pure functions only, so
// the matching rules can be unit-tested without rendering the picker.
import type { TaxonomyTermOut } from '../../api/client';

export const RESULT_CAP = 25;

// 0 = Preferred, 1 = Synonym, 2 = Related. Ties in the term box rank
// Preferred above Synonym above Related.
export type TermLevel = 0 | 1 | 2;
export const LEVEL_LABELS: Record<TermLevel, string> = { 0: 'Preferred', 1: 'Synonym', 2: 'Related' };

export interface TaxonomyEntry {
  code: string;
  group: string;
  term: string;
  level: TermLevel;
  digits: string;
  norm: string;
  words: string[];
}

export interface CodeEntry {
  code: string;
  digits: string;
  group: string;
  preferred: TaxonomyEntry;
  // All terms under this code, Preferred first.
  terms: TaxonomyEntry[];
}

export interface TaxonomyIndex {
  entries: TaxonomyEntry[];
  codes: CodeEntry[];
  byCode: Map<string, TaxonomyEntry[]>;
}

// A missing/unrecognized level (e.g. before the backend reseed lands) is
// treated as a Synonym rather than crashing — it just won't be picked as
// the code's autofilled Preferred term unless it's the only term present.
function levelRank(level: string | null | undefined): TermLevel {
  if (level === 'Preferred') return 0;
  if (level === 'Related') return 2;
  return 1;
}

export function normalize(s: string): string {
  return s
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function compareCodes(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true });
}

export function buildTaxonomyIndex(terms: TaxonomyTermOut[]): TaxonomyIndex {
  const entries: TaxonomyEntry[] = terms
    .filter((t): t is TaxonomyTermOut & { vet_icd_o_code: string } => !!t.vet_icd_o_code)
    .map((t) => {
      const norm = normalize(t.taxonomy_term);
      return {
        code: t.vet_icd_o_code,
        group: t.taxonomy_group,
        term: t.taxonomy_term,
        level: levelRank(t.term_level),
        digits: t.vet_icd_o_code.replace(/\D/g, ''),
        norm,
        words: norm.split(' ').filter(Boolean),
      };
    });

  const byCode = new Map<string, TaxonomyEntry[]>();
  for (const e of entries) {
    const list = byCode.get(e.code);
    if (list) list.push(e);
    else byCode.set(e.code, [e]);
  }
  for (const list of byCode.values()) list.sort((a, b) => a.level - b.level);

  const codes: CodeEntry[] = [...byCode.entries()]
    .map(([code, list]) => ({ code, digits: list[0].digits, group: list[0].group, preferred: list[0], terms: list }))
    .sort((a, b) => compareCodes(a.code, b.code));

  return { entries, codes, byCode };
}

// Optimal string alignment distance (Levenshtein + adjacent transposition).
// Capped: anything more than 2 chars apart in length is reported as 3,
// which is always above the tolerance this module allows.
function osaDistance(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (Math.abs(m - n) > 2) return 3;
  const d: number[][] = [];
  for (let i = 0; i <= m; i++) {
    d.push(new Array<number>(n + 1).fill(0));
    d[i][0] = i;
  }
  for (let j = 0; j <= n; j++) d[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
      }
    }
  }
  return d[m][n];
}

// -1 = no match, 0 = prefix match, 1 = close (typo-tolerant) match.
function wordMatch(queryWord: string, termWord: string): -1 | 0 | 1 {
  if (termWord.startsWith(queryWord)) return 0;
  if (queryWord.length < 4) return -1;
  const allow = queryWord.length >= 8 ? 2 : 1;
  const L = queryWord.length;
  for (const candidate of [termWord, termWord.slice(0, L - 1), termWord.slice(0, L), termWord.slice(0, L + 1)]) {
    if (candidate && osaDistance(queryWord, candidate) <= allow) return 1;
  }
  return -1;
}

export interface CodeSearchResult {
  hits: CodeEntry[];
  total: number;
}

// Matches on digits only, as a prefix. "8010/3", "80103", "8010 3" and
// "8010-3" all resolve to the same query. No typo tolerance — see the
// change request for why.
export function searchCodes(index: TaxonomyIndex, query: string): CodeSearchResult {
  const qd = query.replace(/\D/g, '');
  if (!qd) return { hits: [], total: 0 };
  const hits = index.codes
    .filter((c) => c.digits.startsWith(qd))
    .sort((a, b) => Number(b.digits === qd) - Number(a.digits === qd) || compareCodes(a.code, b.code));
  return { hits, total: hits.length };
}

export interface TermHit {
  entry: TaxonomyEntry;
  fuzzy: boolean;
}

export interface TermSearchResult {
  hits: TermHit[];
  total: number;
}

// Ranking tiers: 0 exact prefix, 1 every query word starts a term word,
// 2 substring anywhere, 3 same as 1 but via typo-tolerant word matches.
// Ties: Preferred -> Synonym -> Related, then shorter term, then A-Z.
export function searchTerms(index: TaxonomyIndex, query: string): TermSearchResult {
  const qn = normalize(query);
  if (!qn) return { hits: [], total: 0 };
  const queryWords = qn.split(' ').filter(Boolean);
  const scored: { entry: TaxonomyEntry; tier: number }[] = [];

  for (const e of index.entries) {
    let tier = -1;
    if (e.norm.startsWith(qn)) {
      tier = 0;
    } else {
      let allWordsMatch = true;
      let anyFuzzy = false;
      for (const qw of queryWords) {
        let best: -1 | 0 | 1 = -1;
        for (const tw of e.words) {
          const r = wordMatch(qw, tw);
          if (r === 0) {
            best = 0;
            break;
          }
          if (r === 1) best = 1;
        }
        if (best === -1) {
          allWordsMatch = false;
          break;
        }
        if (best === 1) anyFuzzy = true;
      }
      if (allWordsMatch) tier = anyFuzzy ? 3 : 1;
      else if (e.norm.includes(qn)) tier = 2;
    }
    if (tier >= 0) scored.push({ entry: e, tier });
  }

  scored.sort(
    (a, b) =>
      a.tier - b.tier ||
      a.entry.level - b.entry.level ||
      a.entry.term.length - b.entry.term.length ||
      a.entry.term.localeCompare(b.entry.term),
  );

  return { hits: scored.map((s) => ({ entry: s.entry, fuzzy: s.tier === 3 })), total: scored.length };
}

export function footerFor(total: number): string {
  return total > RESULT_CAP ? `Showing ${RESULT_CAP} of ${total}. Keep typing to narrow.` : '';
}
