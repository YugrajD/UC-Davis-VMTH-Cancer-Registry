import { act, render, screen, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { renderToString } from 'react-dom/server';
import type { Session, User } from '@supabase/supabase-js';
import { afterEach, describe, expect, it, vi } from 'vitest';

type AuthChangeCallback = (event: string, session: Session | null) => void;
interface TestAuthState {
  user: User | null;
  session: Session | null;
  loading: boolean;
  isAdmin: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  getAccessToken: () => Promise<string | null>;
}

function makeUser(email = 'admin@example.com'): User {
  return {
    id: 'user-1',
    app_metadata: {},
    user_metadata: {},
    aud: 'authenticated',
    created_at: '2026-01-01T00:00:00.000Z',
    email,
  } as User;
}

function makeSession(accessToken = 'token', user = makeUser()): Session {
  return {
    access_token: accessToken,
    refresh_token: 'refresh-token',
    expires_in: 3600,
    token_type: 'bearer',
    user,
  } as Session;
}

async function loadAuthContext({
  configured = true,
  initialSession = null,
  isAdmin = true,
  fetchMeRejects = false,
}: {
  configured?: boolean;
  initialSession?: Session | null;
  isAdmin?: boolean;
  fetchMeRejects?: boolean;
} = {}) {
  vi.resetModules();

  let authChangeCallback: AuthChangeCallback | null = null;
  const unsubscribe = vi.fn();
  const getSession = vi.fn().mockResolvedValue({ data: { session: initialSession } });
  const signInWithPassword = vi.fn().mockResolvedValue({ error: null });
  const signOut = vi.fn().mockResolvedValue({ error: null });
  const onAuthStateChange = vi.fn((callback: AuthChangeCallback) => {
    authChangeCallback = callback;
    return { data: { subscription: { unsubscribe } } };
  });
  const fetchMe = fetchMeRejects
    ? vi.fn().mockRejectedValue(new Error('No admin record'))
    : vi.fn().mockResolvedValue({ email: initialSession?.user.email ?? 'admin@example.com', is_admin: isAdmin });

  vi.doMock('../lib/supabase', () => ({
    supabaseConfigured: configured,
    supabase: {
      auth: {
        getSession,
        signInWithPassword,
        signOut,
        onAuthStateChange,
      },
    },
  }));
  vi.doMock('../api/client', () => ({
    fetchMe,
    ApiError: class ApiError extends Error {
      status: number;
      constructor(status: number, message: string) { super(message); this.status = status; }
    },
  }));

  const authModule = await import('./AuthContext');

  return {
    ...authModule,
    fetchMe,
    getSession,
    signInWithPassword,
    signOut,
    onAuthStateChange,
    unsubscribe,
    emitAuthChange: (session: Session | null) => {
      authChangeCallback?.('SIGNED_IN', session);
    },
  };
}

function StateProbe({
  AuthProvider,
  useAuth,
  children,
}: {
  AuthProvider: ({ children }: { children: ReactNode }) => JSX.Element;
  useAuth: () => TestAuthState;
  children?: (auth: TestAuthState) => void;
}) {
  function Inner() {
    const auth = useAuth();
    children?.(auth);
    return (
      <div>
        <div data-testid="email">{auth.user?.email ?? 'none'}</div>
        <div data-testid="token">{auth.session?.access_token ?? 'none'}</div>
        <div data-testid="loading">{auth.loading ? 'loading' : 'ready'}</div>
        <div data-testid="admin">{auth.isAdmin ? 'admin' : 'not-admin'}</div>
      </div>
    );
  }

  return (
    <AuthProvider>
      <Inner />
    </AuthProvider>
  );
}

afterEach(() => {
  vi.resetModules();
  vi.doUnmock('../lib/supabase');
  vi.doUnmock('../api/client');
});

describe('AuthProvider', () => {
  it('loads the initial Supabase session and admin state', async () => {
    const session = makeSession('initial-token');
    const { AuthProvider, useAuth, fetchMe } = await loadAuthContext({ initialSession: session, isAdmin: true });

    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth} />);

    await waitFor(() => expect(screen.getByTestId('loading')).toHaveTextContent('ready'));
    expect(screen.getByTestId('email')).toHaveTextContent('admin@example.com');
    expect(screen.getByTestId('token')).toHaveTextContent('initial-token');
    await waitFor(() => expect(screen.getByTestId('admin')).toHaveTextContent('admin'));
    expect(fetchMe).toHaveBeenCalledWith('initial-token');
  });

  it('clears user state when there is no session', async () => {
    const { AuthProvider, useAuth, fetchMe } = await loadAuthContext({ initialSession: null });

    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth} />);

    await waitFor(() => expect(screen.getByTestId('loading')).toHaveTextContent('ready'));
    expect(screen.getByTestId('email')).toHaveTextContent('none');
    expect(screen.getByTestId('admin')).toHaveTextContent('not-admin');
    expect(fetchMe).not.toHaveBeenCalled();
  });

  it('keeps admin false when fetchMe fails', async () => {
    const { AuthProvider, useAuth } = await loadAuthContext({
      initialSession: makeSession('token'),
      fetchMeRejects: true,
    });

    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth} />);

    await waitFor(() => expect(screen.getByTestId('loading')).toHaveTextContent('ready'));
    await waitFor(() => expect(screen.getByTestId('admin')).toHaveTextContent('not-admin'));
  });

  it('updates session and rechecks admin on auth state changes', async () => {
    const loginSession = makeSession('login-token', makeUser('user@example.com'));
    const { AuthProvider, useAuth, fetchMe, emitAuthChange } = await loadAuthContext({ initialSession: null });

    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth} />);
    await waitFor(() => expect(screen.getByTestId('loading')).toHaveTextContent('ready'));

    act(() => emitAuthChange(loginSession));

    expect(screen.getByTestId('email')).toHaveTextContent('user@example.com');
    expect(screen.getByTestId('token')).toHaveTextContent('login-token');
    await waitFor(() => expect(screen.getByTestId('admin')).toHaveTextContent('admin'));
    expect(fetchMe).toHaveBeenCalledWith('login-token');
  });

  it('clears admin on auth state changes without a session', async () => {
    const { AuthProvider, useAuth, emitAuthChange } = await loadAuthContext({
      initialSession: makeSession('token'),
      isAdmin: true,
    });

    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth} />);
    await waitFor(() => expect(screen.getByTestId('admin')).toHaveTextContent('admin'));

    act(() => emitAuthChange(null));

    expect(screen.getByTestId('email')).toHaveTextContent('none');
    expect(screen.getByTestId('admin')).toHaveTextContent('not-admin');
  });

  it('signIn calls Supabase and propagates errors', async () => {
    const { AuthProvider, useAuth, signInWithPassword } = await loadAuthContext();
    let auth: TestAuthState | null = null;
    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth}>{value => { auth = value; }}</StateProbe>);
    await waitFor(() => expect(auth).not.toBeNull());
    if (!auth) throw new Error('Auth state was not captured');

    await auth.signIn('user@example.com', 'secret');
    expect(signInWithPassword).toHaveBeenCalledWith({ email: 'user@example.com', password: 'secret' });

    const loginError = new Error('Bad credentials');
    signInWithPassword.mockResolvedValueOnce({ error: loginError });
    await expect(auth.signIn('user@example.com', 'bad')).rejects.toThrow('Bad credentials');
  });

  it('signOut calls Supabase and clears admin', async () => {
    const { AuthProvider, useAuth, signOut } = await loadAuthContext({
      initialSession: makeSession('token'),
      isAdmin: true,
    });
    let auth: TestAuthState | null = null;
    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth}>{value => { auth = value; }}</StateProbe>);
    await waitFor(() => expect(screen.getByTestId('admin')).toHaveTextContent('admin'));
    if (!auth) throw new Error('Auth state was not captured');

    await act(async () => {
      await auth.signOut();
    });

    expect(signOut).toHaveBeenCalled();
    expect(screen.getByTestId('admin')).toHaveTextContent('not-admin');
  });

  it('getAccessToken returns the current session token or null', async () => {
    const { AuthProvider, useAuth, getSession } = await loadAuthContext({
      initialSession: makeSession('token'),
    });
    let auth: TestAuthState | null = null;
    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth}>{value => { auth = value; }}</StateProbe>);
    await waitFor(() => expect(auth).not.toBeNull());
    if (!auth) throw new Error('Auth state was not captured');

    await expect(auth.getAccessToken()).resolves.toBe('token');
    getSession.mockResolvedValueOnce({ data: { session: null } });
    await expect(auth.getAccessToken()).resolves.toBeNull();
  });

  it('useAuth throws outside AuthProvider', async () => {
    const { useAuth } = await loadAuthContext({ configured: false });
    function Consumer() {
      useAuth();
      return null;
    }

    expect(() => renderToString(<Consumer />)).toThrow('useAuth must be used within AuthProvider');
  });
});
