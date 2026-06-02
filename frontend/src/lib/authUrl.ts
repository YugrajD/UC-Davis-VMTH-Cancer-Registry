/**
 * Pure helpers for parsing Supabase auth callbacks out of the URL.
 *
 * Supabase places auth callback data in either the query string (PKCE flow)
 * or the hash fragment (implicit flow).  These helpers normalize both so
 * the AuthContext can call them in a lazy useState initializer.  They MUST
 * stay pure — no DOM writes, no history.replaceState — because React
 * StrictMode runs lazy initializers twice and any side effect would be
 * applied to the URL before the second invocation.
 */

export interface ParsedAuthCallback {
  /** PKCE auth code returned to the app (OAuth or new email flows). */
  code: string | null;
  /** Token hash for verifyOtp — the PKCE-aware email template format. */
  tokenHash: string | null;
  /** Supabase OTP type: 'recovery', 'signup', 'invite', etc. */
  type: string | null;
  /** True when `type === 'recovery'` — used to open the reset modal. */
  isRecovery: boolean;
  /** Human-readable error message, or null when no error was present. */
  error: string | null;
}

const EXPIRED_MESSAGE = 'Your password reset link has expired. Please request a new one.';

function parseError(raw: string): string | null {
  if (!raw.includes('error=')) return null;
  const params = new URLSearchParams(raw);
  const code = params.get('error_code');
  const description = params.get('error_description');
  if (!code && !description) return null;
  if (code === 'otp_expired') return EXPIRED_MESSAGE;
  return description ? description.replace(/\+/g, ' ') : null;
}

/**
 * Parse Supabase auth state from a URL's search and hash components.
 *
 * @param search - The location.search value (with or without the leading `?`).
 * @param hash   - The location.hash value (with or without the leading `#`).
 */
export function parseAuthCallback(search: string, hash: string): ParsedAuthCallback {
  const searchBody = search.startsWith('?') ? search.slice(1) : search;
  const hashBody = hash.startsWith('#') ? hash.slice(1) : hash;

  const params = new URLSearchParams(searchBody);
  const code = params.get('code');
  const tokenHash = params.get('token_hash');
  const type = params.get('type');

  // Error can appear in either source — query string for PKCE callbacks,
  // hash fragment for the older implicit-flow callbacks.
  const error = parseError(searchBody) ?? parseError(hashBody);

  return {
    code,
    tokenHash,
    type,
    isRecovery: type === 'recovery',
    error,
  };
}
