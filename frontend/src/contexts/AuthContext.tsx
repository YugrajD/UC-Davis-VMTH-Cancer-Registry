import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import {
  CognitoUser,
  AuthenticationDetails,
  CognitoUserSession,
} from 'amazon-cognito-identity-js';
import {
  userPool,
  authConfigured,
  authenticationFlowType,
  googleOAuthConfigured,
  googleSignInUrl,
} from '../lib/cognito';
import { parseAuthCallback } from '../lib/authUrl';
import { fetchMe, ApiError } from '../api/client';

export interface AuthUser {
  email: string;
  sub: string;
}

interface AuthState {
  user: AuthUser | null;
  loading: boolean;
  isAdmin: boolean;
  isUploader: boolean;
  isReviewer: boolean;
  authError: string | null;
  googleOAuthConfigured: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  confirmSignUp: (email: string, code: string) => Promise<void>;
  resendConfirmationCode: (email: string) => Promise<void>;
  forgotPassword: (email: string) => Promise<void>;
  confirmForgotPassword: (email: string, code: string, newPassword: string) => Promise<void>;
  signInWithGoogle: () => void;
  signOut: () => void;
  getAccessToken: () => Promise<string | null>;
  clearAuthError: () => void;
}

const AuthContext = createContext<AuthState | undefined>(undefined);

function sessionToUser(session: CognitoUserSession): AuthUser {
  const payload = session.getIdToken().payload;
  return { email: payload.email as string, sub: payload.sub as string };
}

function getCurrentSession(): Promise<CognitoUserSession | null> {
  const cognitoUser = userPool.getCurrentUser();
  if (!cognitoUser) return Promise.resolve(null);
  return new Promise((resolve) => {
    cognitoUser.getSession((err: Error | null, session: CognitoUserSession | null) => {
      resolve(err ? null : session);
    });
  });
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(authConfigured);
  const [isAdmin, setIsAdmin] = useState(false);
  const [isUploader, setIsUploader] = useState(false);
  const [isReviewer, setIsReviewer] = useState(false);
  const [authError, setAuthError] = useState<string | null>(
    () => parseAuthCallback(window.location.search).error,
  );

  useEffect(() => {
    if (!authError) return;
    if (window.location.search.includes('error=')) {
      history.replaceState(null, '', window.location.pathname);
    }
  }, [authError]);

  const refreshRoles = async (idToken: string) => {
    try {
      const me = await fetchMe(idToken);
      setIsAdmin(me.is_admin);
      setIsUploader(me.is_uploader);
      setIsReviewer(me.is_reviewer);
    } catch (err) {
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        setIsAdmin(false);
        setIsUploader(false);
        setIsReviewer(false);
      }
    }
  };

  useEffect(() => {
    if (!authConfigured) return;

    // Hosted UI (Google) redirects back with ?code=... — exchange it for
    // tokens via Cognito's OAuth2 token endpoint.
    const { code, error } = parseAuthCallback(window.location.search);
    if (error) {
      setAuthError(error);
      setLoading(false);
      return;
    }
    if (code) {
      history.replaceState(null, '', window.location.pathname);
      // Hosted-UI code exchange happens server-side via the token endpoint;
      // not implemented for local dev since cognito-local has no Hosted UI.
      setLoading(false);
      return;
    }

    getCurrentSession().then((session) => {
      if (session) {
        setUser(sessionToUser(session));
        refreshRoles(session.getIdToken().getJwtToken());
      }
      setLoading(false);
    });
  }, []);

  const signIn = (email: string, password: string) =>
    new Promise<void>((resolve, reject) => {
      const cognitoUser = new CognitoUser({ Username: email, Pool: userPool });
      cognitoUser.setAuthenticationFlowType(authenticationFlowType);
      const authDetails = new AuthenticationDetails({ Username: email, Password: password });
      cognitoUser.authenticateUser(authDetails, {
        onSuccess: (session) => {
          setUser(sessionToUser(session));
          refreshRoles(session.getIdToken().getJwtToken());
          resolve();
        },
        onFailure: (err) => reject(err),
      });
    });

  const signUp = (email: string, password: string) =>
    new Promise<void>((resolve, reject) => {
      userPool.signUp(email, password, [{ Name: 'email', Value: email }], [], (err) => {
        if (err) reject(err);
        else resolve();
      });
    });

  const confirmSignUp = (email: string, code: string) =>
    new Promise<void>((resolve, reject) => {
      const cognitoUser = new CognitoUser({ Username: email, Pool: userPool });
      cognitoUser.confirmRegistration(code, true, (err) => {
        if (err) reject(err);
        else resolve();
      });
    });

  const resendConfirmationCode = (email: string) =>
    new Promise<void>((resolve, reject) => {
      const cognitoUser = new CognitoUser({ Username: email, Pool: userPool });
      cognitoUser.resendConfirmationCode((err) => {
        if (err) reject(err);
        else resolve();
      });
    });

  const forgotPassword = (email: string) =>
    new Promise<void>((resolve, reject) => {
      const cognitoUser = new CognitoUser({ Username: email, Pool: userPool });
      cognitoUser.forgotPassword({
        onSuccess: () => resolve(),
        onFailure: (err) => reject(err),
      });
    });

  const confirmForgotPassword = (email: string, code: string, newPassword: string) =>
    new Promise<void>((resolve, reject) => {
      const cognitoUser = new CognitoUser({ Username: email, Pool: userPool });
      cognitoUser.confirmPassword(code, newPassword, {
        onSuccess: () => resolve(),
        onFailure: (err) => reject(err),
      });
    });

  const signInWithGoogle = () => {
    if (!googleOAuthConfigured) throw new Error('Google sign-in is not configured');
    window.location.href = googleSignInUrl(window.location.origin);
  };

  const signOut = () => {
    const cognitoUser = userPool.getCurrentUser();
    cognitoUser?.signOut();
    setUser(null);
    setIsAdmin(false);
    setIsUploader(false);
    setIsReviewer(false);
  };

  const getAccessToken = async (): Promise<string | null> => {
    if (!authConfigured) return null;
    const session = await getCurrentSession();
    // The backend verifies the ID token (it carries the `email` claim used
    // for role lookup); getSession() transparently refreshes it if expired.
    return session?.getIdToken().getJwtToken() ?? null;
  };

  const clearAuthError = () => setAuthError(null);

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAdmin,
        isUploader,
        isReviewer,
        authError,
        googleOAuthConfigured,
        signIn,
        signUp,
        confirmSignUp,
        resendConfirmationCode,
        forgotPassword,
        confirmForgotPassword,
        signInWithGoogle,
        signOut,
        getAccessToken,
        clearAuthError,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
