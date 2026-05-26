import { createContext, useContext, useEffect, useState, useCallback, useRef, type ReactNode } from 'react';
import type { User, Session } from '@supabase/supabase-js';
import { supabase, supabaseConfigured } from '../lib/supabase';
import { parseAuthCallback } from '../lib/authUrl';
import { fetchMe, ApiError } from '../api/client';

interface AuthState {
  user: User | null;
  session: Session | null;
  loading: boolean;
  isAdmin: boolean;
  isUploader: boolean;
  isReviewer: boolean;
  passwordRecovery: boolean;
  authError: string | null;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
  getAccessToken: () => Promise<string | null>;
  updatePassword: (newPassword: string) => Promise<void>;
  clearPasswordRecovery: () => void;
  clearAuthError: () => void;
}

const AuthContext = createContext<AuthState | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(supabaseConfigured);
  const [isAdmin, setIsAdmin] = useState(false);
  const [isUploader, setIsUploader] = useState(false);
  const [isReviewer, setIsReviewer] = useState(false);
  const [passwordRecovery, setPasswordRecovery] = useState(false);
  // Read any Supabase error from the URL synchronously so it's available on
  // the very first render with no useEffect needed.  parseAuthCallback is
  // pure (no DOM writes) because React StrictMode runs lazy initializers
  // twice in dev — a side effect here would be applied twice or, worse,
  // applied on the first run and then make the second run return null
  // and overwrite the real state value.  URL cleanup is in a useEffect.
  const [authError, setAuthError] = useState<string | null>(
    () => parseAuthCallback(window.location.search, window.location.hash).error,
  );

  // Strip the error params from the visible URL once we've captured them.
  useEffect(() => {
    if (!authError) return;
    const url = window.location;
    if (url.search.includes('error=') || url.hash.includes('error=')) {
      history.replaceState(null, '', url.pathname);
    }
  }, [authError]);

  // Dedup and backoff refs for refreshRoles
  const inflightRef = useRef(false);
  const backoffRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const refreshRolesRef = useRef<(accessToken: string) => Promise<void>>();

  const refreshRoles = useCallback(async (accessToken: string) => {
    // Skip if a request is already in-flight
    if (inflightRef.current) return;

    // Clear any scheduled retry — a fresh call supersedes it
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = undefined;
    }

    inflightRef.current = true;
    try {
      const me = await fetchMe(accessToken);
      setIsAdmin(me.is_admin);
      setIsUploader(me.is_uploader);
      setIsReviewer(me.is_reviewer);
      backoffRef.current = 0;
    } catch (err) {
      // Only clear roles on explicit auth failures (401/403).
      // Transient errors (5xx, network down) preserve the last-known roles so
      // the admin UI doesn't vanish during a backend restart or blip.
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        setIsAdmin(false);
        setIsUploader(false);
        setIsReviewer(false);
      }
      // Exponential backoff: 2s, 4s, 8s, 16s, 30s cap
      backoffRef.current = Math.min(backoffRef.current + 1, 5);
      const delay = Math.min(1000 * 2 ** backoffRef.current, 30_000);
      retryTimerRef.current = setTimeout(() => refreshRolesRef.current?.(accessToken), delay);
    } finally {
      inflightRef.current = false;
    }
  }, []);

  useEffect(() => {
    refreshRolesRef.current = refreshRoles;
  }, [refreshRoles]);

  useEffect(() => {
    if (!supabaseConfigured) return;

    // Handle email auth callbacks. Two formats, both verified client-side
    // so email scanners that pre-fetch the link can't consume the token:
    //
    //   ?token_hash=...&type=recovery   — PKCE email template (preferred)
    //     verifyOtp is the dedicated entry point for email-confirm flows.
    //   ?code=...                       — OAuth/PKCE callback
    //     exchangeCodeForSession is the dedicated entry point for OAuth.
    const { code, tokenHash, type: emailType, isRecovery } = parseAuthCallback(
      window.location.search,
      window.location.hash,
    );

    if (tokenHash && emailType) {
      history.replaceState(null, '', window.location.pathname);
      supabase.auth
        .verifyOtp({ token_hash: tokenHash, type: emailType as 'recovery' | 'signup' | 'email' | 'invite' | 'email_change' })
        .then(({ error }) => {
          if (error) {
            setAuthError('Could not verify your link. Please request a new one.');
            setLoading(false);
          } else if (isRecovery) {
            setPasswordRecovery(true);
          }
        });
    } else if (code) {
      history.replaceState(null, '', window.location.pathname);
      supabase.auth.exchangeCodeForSession(code).then(({ error }) => {
        if (error) {
          setAuthError('Could not verify your link. Please request a new one.');
          setLoading(false);
        } else if (isRecovery) {
          setPasswordRecovery(true);
        }
      });
    }

    // Get initial session
    supabase.auth.getSession().then(({ data: { session: s } }) => {
      setSession(s);
      setUser(s?.user ?? null);
      if (s?.access_token) {
        refreshRoles(s.access_token);
      }
      setLoading(false);
    });

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, s) => {
      setSession(s);
      setUser(s?.user ?? null);
      if (event === 'PASSWORD_RECOVERY') {
        setPasswordRecovery(true);
      }
      if (s?.access_token) {
        refreshRoles(s.access_token);
      } else {
        setIsAdmin(false);
        setIsUploader(false);
        setIsReviewer(false);
      }
    });

    return () => {
      subscription.unsubscribe();
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    };
  }, [refreshRoles]);

  const signIn = async (email: string, password: string) => {
    if (!supabaseConfigured) throw new Error('Auth is not configured');
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw error;
  };

  const signUp = async (email: string, password: string) => {
    if (!supabaseConfigured) throw new Error('Auth is not configured');
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) throw error;
  };

  const signInWithGoogle = async () => {
    if (!supabaseConfigured) throw new Error('Auth is not configured');
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: window.location.origin },
    });
    if (error) throw error;
  };

  const signOut = async () => {
    if (!supabaseConfigured) return;
    await supabase.auth.signOut();
    setIsAdmin(false);
    setIsUploader(false);
    setIsReviewer(false);
  };

  const getAccessToken = async (): Promise<string | null> => {
    if (!supabaseConfigured) return null;
    const { data } = await supabase.auth.getSession();
    if (!data.session) return null;
    // Proactively refresh if the access token expires within 60 seconds so
    // the backend never receives an expired token (which triggers the auth
    // failure rate limiter and causes 429s on subsequent valid requests).
    const expiresAt = data.session.expires_at;
    if (expiresAt !== undefined && Date.now() / 1000 >= expiresAt - 60) {
      const { data: refreshed } = await supabase.auth.refreshSession();
      return refreshed.session?.access_token ?? null;
    }
    return data.session.access_token;
  };

  const updatePassword = async (newPassword: string) => {
    if (!supabaseConfigured) throw new Error('Auth is not configured');
    const { error } = await supabase.auth.updateUser({ password: newPassword });
    if (error) throw error;
    setPasswordRecovery(false);
  };

  const clearPasswordRecovery = () => setPasswordRecovery(false);
  const clearAuthError = () => setAuthError(null);

  return (
    <AuthContext.Provider value={{ user, session, loading, isAdmin, isUploader, isReviewer, passwordRecovery, authError, signIn, signUp, signInWithGoogle, signOut, getAccessToken, updatePassword, clearPasswordRecovery, clearAuthError }}>
      {children}
    </AuthContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
