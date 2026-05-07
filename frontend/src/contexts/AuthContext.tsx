import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react';
import type { User, Session } from '@supabase/supabase-js';
import { supabase, supabaseConfigured } from '../lib/supabase';
import { fetchMe } from '../api/client';

interface AuthState {
  user: User | null;
  session: Session | null;
  loading: boolean;
  isAdmin: boolean;
  isUploader: boolean;
  isReviewer: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  getAccessToken: () => Promise<string | null>;
}

const AuthContext = createContext<AuthState | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(supabaseConfigured);
  const [isAdmin, setIsAdmin] = useState(false);
  const [isUploader, setIsUploader] = useState(false);
  const [isReviewer, setIsReviewer] = useState(false);

  const refreshRoles = useCallback(async (accessToken: string) => {
    try {
      const me = await fetchMe(accessToken);
      setIsAdmin(me.is_admin);
      setIsUploader(me.is_uploader);
      setIsReviewer(me.is_reviewer);
    } catch {
      setIsAdmin(false);
      setIsUploader(false);
      setIsReviewer(false);
    }
  }, []);

  useEffect(() => {
    if (!supabaseConfigured) return;

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
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, s) => {
      setSession(s);
      setUser(s?.user ?? null);
      if (s?.access_token) {
        refreshRoles(s.access_token);
      } else {
        setIsAdmin(false);
        setIsUploader(false);
        setIsReviewer(false);
      }
    });

    return () => subscription.unsubscribe();
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
    return data.session?.access_token ?? null;
  };

  return (
    <AuthContext.Provider value={{ user, session, loading, isAdmin, isUploader, isReviewer, signIn, signUp, signOut, getAccessToken }}>
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
