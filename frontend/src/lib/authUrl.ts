/**
 * Pure helpers for parsing Cognito Hosted UI OAuth callbacks out of the URL.
 *
 * Cognito's Hosted UI (Google sign-in) redirects back with a standard
 * OAuth2 authorization-code response: `?code=...` on success, or
 * `?error=...&error_description=...` on failure. Email sign-up confirmation
 * and password reset use codes entered in-app instead of links, so they
 * don't appear here. This MUST stay pure — no DOM writes, no
 * history.replaceState — because React StrictMode runs lazy initializers
 * twice and any side effect would be applied to the URL before the second
 * invocation.
 */

export interface ParsedAuthCallback {
  /** OAuth2 authorization code from the Hosted UI redirect. */
  code: string | null;
  /** Human-readable error message, or null when no error was present. */
  error: string | null;
}

/**
 * Parse a Cognito Hosted UI OAuth callback from a URL's search component.
 *
 * @param search - The location.search value (with or without the leading `?`).
 */
export function parseAuthCallback(search: string): ParsedAuthCallback {
  const body = search.startsWith('?') ? search.slice(1) : search;
  const params = new URLSearchParams(body);
  const code = params.get('code');

  const errorCode = params.get('error');
  const description = params.get('error_description');
  const error = errorCode ? (description ? description.replace(/\+/g, ' ') : errorCode) : null;

  return { code, error };
}
