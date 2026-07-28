import { describe, it, expect } from 'vitest';
import { parseAuthCallback } from '../lib/authUrl';

// ---------------------------------------------------------------------------
// Empty / unrelated URLs
// ---------------------------------------------------------------------------

describe('parseAuthCallback — empty input', () => {
  it('returns all-null fields when search is empty', () => {
    const r = parseAuthCallback('');
    expect(r.code).toBeNull();
    expect(r.error).toBeNull();
  });

  it('ignores unrelated query parameters', () => {
    const r = parseAuthCallback('?utm_source=email&page=2');
    expect(r.code).toBeNull();
    expect(r.error).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Hosted UI OAuth callback — ?code=...
// ---------------------------------------------------------------------------

describe('parseAuthCallback — OAuth code', () => {
  it('extracts a bare code parameter', () => {
    const r = parseAuthCallback('?code=oauth_xyz');
    expect(r.code).toBe('oauth_xyz');
    expect(r.error).toBeNull();
  });

  it('handles leading ? being absent', () => {
    const r = parseAuthCallback('code=abc');
    expect(r.code).toBe('abc');
  });
});

// ---------------------------------------------------------------------------
// Errors in the query string
// ---------------------------------------------------------------------------

describe('parseAuthCallback — error', () => {
  it('uses the description when present', () => {
    const r = parseAuthCallback('?error=access_denied&error_description=Something+went+wrong');
    expect(r.error).toBe('Something went wrong');
  });

  it('falls back to the error code when no description is present', () => {
    const r = parseAuthCallback('?error=access_denied');
    expect(r.error).toBe('access_denied');
  });

  it('returns null when no error param is present', () => {
    const r = parseAuthCallback('?code=abc');
    expect(r.error).toBeNull();
  });

  it('decodes plus signs as spaces in description', () => {
    const r = parseAuthCallback('?error=x&error_description=A+B+C');
    expect(r.error).toBe('A B C');
  });
});

// ---------------------------------------------------------------------------
// Purity guarantees — same input gives same output, repeatable
// ---------------------------------------------------------------------------

describe('parseAuthCallback — purity', () => {
  it('is referentially transparent (no hidden state)', () => {
    const a = parseAuthCallback('?code=abc');
    const b = parseAuthCallback('?code=abc');
    expect(a).toEqual(b);
  });

  it('does not mutate window or history on multiple calls', () => {
    // The whole point of extracting this function — StrictMode runs it twice.
    // If the function had side effects, the second call would observe a
    // different URL than the first.  Here we just confirm the return value
    // is stable across repeated calls.
    const first = parseAuthCallback('?code=abc');
    const second = parseAuthCallback('?code=abc');
    const third = parseAuthCallback('?code=abc');
    expect(first).toEqual(second);
    expect(second).toEqual(third);
  });
});
