import { describe, it, expect } from 'vitest';
import { parseAuthCallback } from '../lib/authUrl';

// ---------------------------------------------------------------------------
// Empty / unrelated URLs
// ---------------------------------------------------------------------------

describe('parseAuthCallback — empty input', () => {
  it('returns all-null fields when both search and hash are empty', () => {
    const r = parseAuthCallback('', '');
    expect(r.code).toBeNull();
    expect(r.tokenHash).toBeNull();
    expect(r.type).toBeNull();
    expect(r.isRecovery).toBe(false);
    expect(r.error).toBeNull();
  });

  it('ignores unrelated query parameters', () => {
    const r = parseAuthCallback('?utm_source=email&page=2', '');
    expect(r.code).toBeNull();
    expect(r.tokenHash).toBeNull();
    expect(r.error).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// PKCE email template — ?token_hash=...&type=recovery
// ---------------------------------------------------------------------------

describe('parseAuthCallback — PKCE email (token_hash)', () => {
  it('extracts token_hash and type=recovery', () => {
    const r = parseAuthCallback(
      '?token_hash=pkce_abc123&type=recovery',
      '',
    );
    expect(r.tokenHash).toBe('pkce_abc123');
    expect(r.type).toBe('recovery');
    expect(r.isRecovery).toBe(true);
    expect(r.error).toBeNull();
  });

  it('handles leading ? being absent', () => {
    const r = parseAuthCallback('token_hash=abc&type=recovery', '');
    expect(r.tokenHash).toBe('abc');
    expect(r.isRecovery).toBe(true);
  });

  it('marks type=signup as non-recovery', () => {
    const r = parseAuthCallback('?token_hash=abc&type=signup', '');
    expect(r.type).toBe('signup');
    expect(r.isRecovery).toBe(false);
  });

  it('handles type=email_change as non-recovery', () => {
    const r = parseAuthCallback('?token_hash=abc&type=email_change', '');
    expect(r.isRecovery).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// OAuth/PKCE — ?code=...
// ---------------------------------------------------------------------------

describe('parseAuthCallback — OAuth/PKCE code', () => {
  it('extracts a bare code parameter', () => {
    const r = parseAuthCallback('?code=oauth_xyz', '');
    expect(r.code).toBe('oauth_xyz');
    expect(r.tokenHash).toBeNull();
    expect(r.isRecovery).toBe(false);
  });

  it('combines code with a recovery type marker', () => {
    // Legacy redirectTo flow that included ?type=recovery as a marker.
    const r = parseAuthCallback('?code=xyz&type=recovery', '');
    expect(r.code).toBe('xyz');
    expect(r.type).toBe('recovery');
    expect(r.isRecovery).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Errors in the query string (PKCE flow)
// ---------------------------------------------------------------------------

describe('parseAuthCallback — query-string error', () => {
  it('returns the expired-link message for otp_expired', () => {
    const r = parseAuthCallback(
      '?error=access_denied&error_code=otp_expired&error_description=Email+link+is+invalid+or+has+expired',
      '',
    );
    expect(r.error).toBe('Your password reset link has expired. Please request a new one.');
  });

  it('falls back to the description for unknown error codes', () => {
    const r = parseAuthCallback(
      '?error=server_error&error_code=internal&error_description=Something+went+wrong',
      '',
    );
    expect(r.error).toBe('Something went wrong');
  });

  it('decodes plus signs as spaces in description', () => {
    const r = parseAuthCallback(
      '?error=x&error_code=other&error_description=A+B+C',
      '',
    );
    expect(r.error).toBe('A B C');
  });

  it('ignores error= with no code or description', () => {
    const r = parseAuthCallback('?error=', '');
    expect(r.error).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Errors in the hash fragment (implicit flow)
// ---------------------------------------------------------------------------

describe('parseAuthCallback — hash-fragment error', () => {
  it('extracts otp_expired from the hash', () => {
    const r = parseAuthCallback(
      '',
      '#error=access_denied&error_code=otp_expired&error_description=Email+link+is+invalid+or+has+expired',
    );
    expect(r.error).toBe('Your password reset link has expired. Please request a new one.');
  });

  it('handles leading # being absent', () => {
    const r = parseAuthCallback('', 'error=access_denied&error_code=otp_expired&error_description=x');
    expect(r.error).toBe('Your password reset link has expired. Please request a new one.');
  });

  it('prefers the query-string error over the hash error', () => {
    // Both shouldn't normally co-occur, but if they do, search wins.
    const r = parseAuthCallback(
      '?error=x&error_code=otp_expired',
      '#error=x&error_code=other&error_description=Hash+error',
    );
    expect(r.error).toBe('Your password reset link has expired. Please request a new one.');
  });
});

// ---------------------------------------------------------------------------
// Combined recovery + error (the case that originally broke in StrictMode)
// ---------------------------------------------------------------------------

describe('parseAuthCallback — recovery error scenario', () => {
  it('parses error in query string even when type=recovery is present', () => {
    const r = parseAuthCallback(
      '?error=access_denied&error_code=otp_expired&error_description=Email+link+is+invalid+or+has+expired&type=recovery',
      '',
    );
    expect(r.error).toBe('Your password reset link has expired. Please request a new one.');
    expect(r.type).toBe('recovery');
    expect(r.isRecovery).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Purity guarantees — same input gives same output, repeatable
// ---------------------------------------------------------------------------

describe('parseAuthCallback — purity', () => {
  it('is referentially transparent (no hidden state)', () => {
    const a = parseAuthCallback('?code=abc', '');
    const b = parseAuthCallback('?code=abc', '');
    expect(a).toEqual(b);
  });

  it('does not mutate window or history on multiple calls', () => {
    // The whole point of extracting this function — StrictMode runs it twice.
    // If the function had side effects, the second call would observe a
    // different URL than the first.  Here we just confirm the return value
    // is stable across repeated calls.
    const first = parseAuthCallback('?token_hash=abc&type=recovery', '');
    const second = parseAuthCallback('?token_hash=abc&type=recovery', '');
    const third = parseAuthCallback('?token_hash=abc&type=recovery', '');
    expect(first).toEqual(second);
    expect(second).toEqual(third);
  });
});
