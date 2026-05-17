import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { supabase, supabaseConfigured } from '../../lib/supabase';

interface LoginModalProps {
  onClose: () => void;
}

export function LoginModal({ onClose }: LoginModalProps) {
  const { signIn, signUp, signInWithGoogle, authError, clearAuthError } = useAuth();
  // If there's a context-level auth error (e.g. expired reset link), start in
  // forgot-password mode so the user can immediately request a new one.
  const [mode, setMode] = useState<'signin' | 'signup' | 'forgot'>(authError ? 'forgot' : 'signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(authError ?? null);
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [signupSuccess, setSignupSuccess] = useState(false);
  const [resetSent, setResetSent] = useState(false);

  // Clear the context-level error when the modal closes.
  const handleClose = () => {
    clearAuthError();
    onClose();
  };

  const handleGoogleSignIn = async () => {
    setError(null);
    setGoogleLoading(true);
    try {
      await signInWithGoogle();
      // Page will redirect to Google — no onClose() needed
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Google sign-in failed');
      setGoogleLoading(false);
    }
  };

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { clearAuthError(); onClose(); } };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose, clearAuthError]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (mode === 'signup' && password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);
    try {
      if (mode === 'signin') {
        await signIn(email, password);
        handleClose();
      } else if (mode === 'signup') {
        await signUp(email, password);
        setSignupSuccess(true);
      } else {
        // The ?type=recovery marker lets AuthContext distinguish a password
        // reset code from an ordinary sign-in code after exchangeCodeForSession.
        const { error: resetError } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: `${window.location.origin}/?type=recovery`,
        });
        if (resetError) throw resetError;
        setResetSent(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : `${mode === 'signin' ? 'Sign in' : mode === 'signup' ? 'Sign up' : 'Password reset'} failed`);
    } finally {
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setMode(mode === 'signin' ? 'signup' : 'signin');
    setError(null);
    setSignupSuccess(false);
    setConfirmPassword('');
  };

  const goToForgot = () => {
    setMode('forgot');
    setError(null);
    setPassword('');
    setConfirmPassword('');
  };

  const backToSignIn = () => {
    setMode('signin');
    setError(null);
    setSignupSuccess(false);
    setResetSent(false);
    setConfirmPassword('');
    clearAuthError();
  };

  const modalTitle = mode === 'signin' ? 'Sign In' : mode === 'signup' ? 'Create Account' : 'Reset Password';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={handleClose}>
      <div className="absolute inset-0 bg-black/50" />
      <div
        className="relative bg-white rounded-xl shadow-2xl w-full max-w-md p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">
            {modalTitle}
          </h2>
          <button
            onClick={handleClose}
            className="p-1.5 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {!supabaseConfigured ? (
          <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
            <p className="text-sm font-medium text-yellow-800 mb-1">Auth not configured</p>
            <p className="text-sm text-yellow-700">
              Set <code className="bg-yellow-100 px-1 rounded text-xs">VITE_SUPABASE_URL</code> and{' '}
              <code className="bg-yellow-100 px-1 rounded text-xs">VITE_SUPABASE_ANON_KEY</code> environment
              variables, then restart the frontend.
            </p>
          </div>
        ) : signupSuccess ? (
          <div className="space-y-4">
            <div className="bg-green-50 border border-green-200 rounded-md p-4">
              <p className="text-sm font-medium text-green-800 mb-1">Account created</p>
              <p className="text-sm text-green-700">
                Check your email for a confirmation link. Once confirmed, you can sign in.
              </p>
              <p className="text-xs text-green-600 mt-2">
                If you don&apos;t see it, check your spam or junk folder.
              </p>
            </div>
            <button
              onClick={backToSignIn}
              className="w-full py-2.5 bg-[var(--color-teal)] text-white text-sm font-semibold rounded-md hover:bg-[var(--color-teal-dark)] transition-colors"
            >
              Back to Sign In
            </button>
          </div>
        ) : resetSent ? (
          <div className="space-y-4">
            <div className="bg-green-50 border border-green-200 rounded-md p-4">
              <p className="text-sm font-medium text-green-800 mb-1">Reset email sent</p>
              <p className="text-sm text-green-700">
                Check your inbox for a password reset link.
              </p>
              <p className="text-xs text-green-600 mt-2">
                If you don&apos;t see it, check your spam or junk folder.
              </p>
            </div>
            <button
              onClick={backToSignIn}
              className="w-full py-2.5 bg-[var(--color-teal)] text-white text-sm font-semibold rounded-md hover:bg-[var(--color-teal-dark)] transition-colors"
            >
              Back to Sign In
            </button>
          </div>
        ) : mode === 'forgot' ? (
          <form onSubmit={handleSubmit} className="space-y-4">
            <p className="text-sm text-gray-600">
              Enter your email and we&apos;ll send you a link to reset your password.
            </p>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                placeholder="you@example.com"
              />
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-md p-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 bg-[var(--color-teal)] text-white text-sm font-semibold rounded-md hover:bg-[var(--color-teal-dark)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Sending...' : 'Send Reset Link'}
            </button>

            <p className="text-center text-sm text-gray-600">
              <button type="button" onClick={backToSignIn} className="text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium">
                Back to Sign In
              </button>
            </p>
          </form>
        ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
              placeholder="you@example.com"
            />
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-sm font-medium text-gray-700">Password</label>
              {mode === 'signin' && (
                <button
                  type="button"
                  onClick={goToForgot}
                  className="text-xs text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium"
                >
                  Forgot password?
                </button>
              )}
            </div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
              placeholder="Password"
            />
          </div>

          {mode === 'signup' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                minLength={6}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                placeholder="Confirm password"
              />
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 bg-[var(--color-teal)] text-white text-sm font-semibold rounded-md hover:bg-[var(--color-teal-dark)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading
              ? (mode === 'signin' ? 'Signing in...' : 'Creating account...')
              : (mode === 'signin' ? 'Sign In' : 'Create Account')
            }
          </button>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200" />
            </div>
            <div className="relative flex justify-center text-xs">
              <span className="bg-white px-2 text-gray-400">or</span>
            </div>
          </div>

          <button
            type="button"
            onClick={handleGoogleSignIn}
            disabled={googleLoading || loading}
            className="w-full flex items-center justify-center gap-3 px-4 py-2.5 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {googleLoading ? (
              <div className="w-4 h-4 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4" viewBox="0 0 24 24">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
              </svg>
            )}
            Continue with Google
          </button>

          <p className="text-center text-sm text-gray-600">
            {mode === 'signin' ? (
              <>Don&apos;t have an account?{' '}
                <button type="button" onClick={toggleMode} className="text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium">
                  Sign up
                </button>
              </>
            ) : (
              <>Already have an account?{' '}
                <button type="button" onClick={toggleMode} className="text-[var(--color-teal)] hover:text-[var(--color-teal-dark)] font-medium">
                  Sign in
                </button>
              </>
            )}
          </p>
        </form>
        )}
      </div>
    </div>
  );
}
