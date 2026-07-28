import { act, render, screen, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { renderToString } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';

interface TestAuthState {
  user: { email: string; sub: string } | null;
  loading: boolean;
  isAdmin: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => void;
  getAccessToken: () => Promise<string | null>;
}

function makeIdTokenPayload(email = 'admin@example.com', sub = 'user-1') {
  return { email, sub };
}

function makeSession(idTokenPayload = makeIdTokenPayload(), jwt = 'token') {
  return {
    getIdToken: () => ({ payload: idTokenPayload, getJwtToken: () => jwt }),
  };
}

async function loadAuthContext({
  configured = true,
  initialSession = null,
  isAdmin = true,
  fetchMeRejects = false,
}: {
  configured?: boolean;
  initialSession?: ReturnType<typeof makeSession> | null;
  isAdmin?: boolean;
  fetchMeRejects?: boolean;
} = {}) {
  vi.resetModules();

  const getSession = vi.fn((callback: (err: Error | null, session: unknown) => void) => {
    callback(null, initialSession);
  });
  const signOut = vi.fn();
  const getCurrentUser = vi.fn(() => (initialSession ? { getSession, signOut } : null));

  let capturedAuthenticateUser: ((details: unknown, callbacks: {
    onSuccess: (session: unknown) => void;
    onFailure: (err: Error) => void;
  }) => void) | null = null;

  function CognitoUser() {
    return {
      authenticateUser: (details: unknown, callbacks: { onSuccess: (s: unknown) => void; onFailure: (e: Error) => void }) => {
        capturedAuthenticateUser?.(details, callbacks);
      },
      getSession,
      signOut,
    };
  }

  const userPool = { getCurrentUser };

  const fetchMe = fetchMeRejects
    ? vi.fn().mockRejectedValue(new Error('No admin record'))
    : vi.fn().mockResolvedValue({ email: initialSession?.getIdToken().payload.email ?? 'admin@example.com', is_admin: isAdmin });

  vi.doMock('../lib/cognito', () => ({
    authConfigured: configured,
    googleOAuthConfigured: false,
    userPool,
    googleSignInUrl: () => 'https://example.com',
  }));
  vi.doMock('amazon-cognito-identity-js', () => ({
    CognitoUser,
    AuthenticationDetails: function AuthenticationDetails(d: unknown) { return d; },
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
    signOut,
    setAuthenticateUserHandler: (fn: typeof capturedAuthenticateUser) => { capturedAuthenticateUser = fn; },
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
  vi.doUnmock('../lib/cognito');
  vi.doUnmock('amazon-cognito-identity-js');
  vi.doUnmock('../api/client');
});

describe('AuthProvider', () => {
  it('loads the initial Cognito session and admin state', async () => {
    const session = makeSession(makeIdTokenPayload('admin@example.com'), 'initial-token');
    const { AuthProvider, useAuth, fetchMe } = await loadAuthContext({ initialSession: session, isAdmin: true });

    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth} />);

    await waitFor(() => expect(screen.getByTestId('loading')).toHaveTextContent('ready'));
    expect(screen.getByTestId('email')).toHaveTextContent('admin@example.com');
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
      initialSession: makeSession(),
      fetchMeRejects: true,
    });

    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth} />);

    await waitFor(() => expect(screen.getByTestId('loading')).toHaveTextContent('ready'));
    await waitFor(() => expect(screen.getByTestId('admin')).toHaveTextContent('not-admin'));
  });

  it('signIn authenticates against Cognito and updates state', async () => {
    const { AuthProvider, useAuth, setAuthenticateUserHandler, fetchMe } = await loadAuthContext({ initialSession: null });
    setAuthenticateUserHandler((_details, callbacks) => {
      callbacks.onSuccess(makeSession(makeIdTokenPayload('user@example.com'), 'login-token'));
    });

    let auth: TestAuthState | null = null;
    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth}>{value => { auth = value; }}</StateProbe>);
    await waitFor(() => expect(auth).not.toBeNull());
    if (!auth) throw new Error('Auth state was not captured');

    await act(async () => {
      await auth!.signIn('user@example.com', 'secret');
    });

    expect(screen.getByTestId('email')).toHaveTextContent('user@example.com');
    await waitFor(() => expect(fetchMe).toHaveBeenCalledWith('login-token'));
  });

  it('signIn propagates errors', async () => {
    const { AuthProvider, useAuth, setAuthenticateUserHandler } = await loadAuthContext({ initialSession: null });
    setAuthenticateUserHandler((_details, callbacks) => {
      callbacks.onFailure(new Error('Bad credentials'));
    });

    let auth: TestAuthState | null = null;
    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth}>{value => { auth = value; }}</StateProbe>);
    await waitFor(() => expect(auth).not.toBeNull());
    if (!auth) throw new Error('Auth state was not captured');

    await expect(auth.signIn('user@example.com', 'bad')).rejects.toThrow('Bad credentials');
  });

  it('signOut clears session and admin state', async () => {
    const { AuthProvider, useAuth, signOut } = await loadAuthContext({
      initialSession: makeSession(),
      isAdmin: true,
    });
    let auth: TestAuthState | null = null;
    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth}>{value => { auth = value; }}</StateProbe>);
    await waitFor(() => expect(screen.getByTestId('admin')).toHaveTextContent('admin'));
    if (!auth) throw new Error('Auth state was not captured');

    act(() => { auth!.signOut(); });

    expect(signOut).toHaveBeenCalled();
    expect(screen.getByTestId('admin')).toHaveTextContent('not-admin');
  });

  it('getAccessToken returns the current ID token or null', async () => {
    const { AuthProvider, useAuth } = await loadAuthContext({
      initialSession: makeSession(makeIdTokenPayload(), 'the-token'),
    });
    let auth: TestAuthState | null = null;
    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth}>{value => { auth = value; }}</StateProbe>);
    await waitFor(() => expect(auth).not.toBeNull());
    if (!auth) throw new Error('Auth state was not captured');

    await expect(auth.getAccessToken()).resolves.toBe('the-token');
  });

  it('getAccessToken returns null with no session', async () => {
    const { AuthProvider, useAuth } = await loadAuthContext({ initialSession: null });
    let auth: TestAuthState | null = null;
    render(<StateProbe AuthProvider={AuthProvider} useAuth={useAuth}>{value => { auth = value; }}</StateProbe>);
    await waitFor(() => expect(auth).not.toBeNull());
    if (!auth) throw new Error('Auth state was not captured');

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
