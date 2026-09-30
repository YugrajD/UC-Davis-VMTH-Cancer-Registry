import { describe, expect, it } from 'vitest';
import type { TaxonomyTermOut } from '../../api/client';
import { buildTaxonomyIndex, footerFor, RESULT_CAP, searchCodes, searchTerms } from './taxonomyMatch';
import fixture from './__fixtures__/taxonomyTerms.fixture.json';

// The real Vet-ICD-O-canine-1 list (845 rows), parsed from ml/taxonomy/labels.csv.
const terms = fixture as TaxonomyTermOut[];
const index = buildTaxonomyIndex(terms);

describe('buildTaxonomyIndex', () => {
  it('groups all 845 entries under 534 distinct codes', () => {
    expect(index.entries).toHaveLength(845);
    expect(index.codes).toHaveLength(534);
  });

  it('sorts each code group Preferred first', () => {
    const siblings = index.byCode.get('8010/3')!;
    expect(siblings.map((e) => e.term)).toEqual(['Carcinoma, NOS', 'Epithelial tumor, malignant']);
    expect(siblings[0].level).toBe(0);
  });
});

describe('searchCodes', () => {
  it('matches digits as a prefix regardless of punctuation', () => {
    for (const q of ['8010/3', '80103', '8010 3', '8010-3']) {
      const { hits } = searchCodes(index, q);
      expect(hits[0].code).toBe('8010/3');
    }
  });

  it('returns nothing for an empty or letters-only query', () => {
    expect(searchCodes(index, '').hits).toEqual([]);
    expect(searchCodes(index, 'abc').hits).toEqual([]);
  });

  it('ranks an exact digit match first, then numeric code order', () => {
    const { hits } = searchCodes(index, '9120');
    expect(hits.map((c) => c.code)).toEqual([
      '9120.0/0',
      '9120.1/0',
      '9120.1/3',
      '9120.2/0',
      '9120.2/3',
      '9120.3/0',
      '9120/0',
      '9120/3',
    ]);
  });
});

describe('searchTerms', () => {
  it('ranks an exact prefix match first', () => {
    const { hits } = searchTerms(index, 'hemangiosarcoma');
    expect(hits[0].entry.term).toBe('Hemangiosarcoma, NOS');
    expect(hits[0].fuzzy).toBe(false);
  });

  it('applies typo tolerance and labels the result a close match', () => {
    const { hits } = searchTerms(index, 'hemangiosarcma');
    expect(hits[0].entry.code).toBe('9120/3');
    expect(hits[0].entry.term).toBe('Hemangiosarcoma, NOS');
    expect(hits[0].fuzzy).toBe(true);
  });

  it('does not let a close match outrank a real prefix match', () => {
    const { hits } = searchTerms(index, 'carcinoma');
    const carcinoidIdx = hits.findIndex((h) => h.entry.term.toLowerCase().includes('carcinoid'));
    const carcinomaIdx = hits.findIndex((h) => h.entry.term === 'Carcinoma, NOS');
    expect(carcinomaIdx).toBeGreaterThanOrEqual(0);
    if (carcinoidIdx >= 0) expect(carcinomaIdx).toBeLessThan(carcinoidIdx);
  });

  it('returns 82 hits for "lymphoma", above the 25-row cap', () => {
    const { total } = searchTerms(index, 'lymphoma');
    expect(total).toBe(82);
    expect(footerFor(total)).toBe(`Showing ${RESULT_CAP} of 82. Keep typing to narrow.`);
  });

  it('finds a term listed under two codes', () => {
    const matches = index.entries.filter((e) => e.norm === 'papillary adenocarcinoma');
    expect(matches.map((e) => e.code)).toEqual(['8050/3', '8260/3']);
  });

  it('is case- and punctuation-insensitive', () => {
    const { hits } = searchTerms(index, 'HEMANGIOSARCOMA, NOS');
    expect(hits[0].entry.term).toBe('Hemangiosarcoma, NOS');
  });
});
