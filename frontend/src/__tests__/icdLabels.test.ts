import { describe, expect, it } from 'vitest';
import { ICD_LABELS, formatIcdOptionLabel } from '../data/icdLabels';

describe('ICD_LABELS', () => {
  it('includes Vet-ICD-O morphology code labels', () => {
    expect(ICD_LABELS).toEqual(
      expect.arrayContaining([
        { code: '8000/3', term: 'Neoplasm, malignant' },
        { code: '8070/3', term: 'Squamous cell carcinoma, NOS' },
      ]),
    );
  });

  it('formats dropdown labels as code followed by diagnosis in parentheses', () => {
    expect(formatIcdOptionLabel({ code: '8000/3', term: 'Neoplasm, malignant' }))
      .toBe('8000/3 (Neoplasm, malignant)');
  });
});
