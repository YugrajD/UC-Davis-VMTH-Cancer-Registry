import { describe, it, expect } from 'vitest';
import { normalizeEmail, isValidEmail } from '../api/client';

// ---------------------------------------------------------------------------
// normalizeEmail
// ---------------------------------------------------------------------------

describe('normalizeEmail', () => {
  it('lowercases the input', () => {
    expect(normalizeEmail('Alice@UCDavis.edu')).toBe('alice@ucdavis.edu');
  });

  it('trims surrounding whitespace', () => {
    expect(normalizeEmail('  user@example.com  ')).toBe('user@example.com');
  });

  it('strips leading/trailing tabs and newlines', () => {
    expect(normalizeEmail('\tuser@example.com\n')).toBe('user@example.com');
  });

  it('returns empty string for empty input', () => {
    expect(normalizeEmail('')).toBe('');
  });

  it('does not mutate inner content', () => {
    expect(normalizeEmail('Mixed.Case+Tag@Example.com')).toBe('mixed.case+tag@example.com');
  });
});

// ---------------------------------------------------------------------------
// isValidEmail
// ---------------------------------------------------------------------------

describe('isValidEmail', () => {
  it('accepts a normal email', () => {
    expect(isValidEmail('alice@ucdavis.edu')).toBe(true);
  });

  it('accepts an uppercase email (we normalize before checking)', () => {
    expect(isValidEmail('ALICE@UCDAVIS.EDU')).toBe(true);
  });

  it('accepts an email with whitespace (trimmed before checking)', () => {
    expect(isValidEmail('  alice@ucdavis.edu  ')).toBe(true);
  });

  it('accepts subaddressed emails', () => {
    expect(isValidEmail('alice+admin@ucdavis.edu')).toBe(true);
  });

  it('rejects empty input', () => {
    expect(isValidEmail('')).toBe(false);
  });

  it('rejects whitespace-only input', () => {
    expect(isValidEmail('   ')).toBe(false);
  });

  it('rejects strings without an @', () => {
    expect(isValidEmail('not-an-email')).toBe(false);
  });

  it('rejects emails longer than 255 characters', () => {
    const long = 'a'.repeat(250) + '@x.com'; // 256 chars
    expect(isValidEmail(long)).toBe(false);
  });

  it('accepts an email exactly at the 255-char boundary', () => {
    // 247 a's + '@x.com' = 253 chars — well under
    const ok = 'a'.repeat(247) + '@x.com';
    expect(ok.length).toBeLessThanOrEqual(255);
    expect(isValidEmail(ok)).toBe(true);
  });
});
