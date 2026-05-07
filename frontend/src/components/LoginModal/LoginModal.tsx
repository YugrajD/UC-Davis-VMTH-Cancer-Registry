import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { supabaseConfigured } from '../../lib/supabase';

interface LoginModalProps {
  onClose: () => void;
}

export function LoginModal({ onClose }: LoginModalProps) {
  const { signIn, signUp } = useAuth();
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [signupSuccess, setSignupSuccess] = useState(false);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

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
        onClose();
      } else {
        await signUp(email, password);
        setSignupSuccess(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : `${mode === 'signin' ? 'Sign in' : 'Sign up'} failed`);
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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/50" />
      <div
        className="relative bg-white rounded-xl shadow-2xl w-full max-w-md p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">
            {mode === 'signin' ? 'Sign In' : 'Create Account'}
          </h2>
          <button
            onClick={onClose}
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
            </div>
            <button
              onClick={toggleMode}
              className="w-full py-2.5 bg-[var(--color-teal)] text-white text-sm font-semibold rounded-md hover:bg-[var(--color-teal-dark)] transition-colors"
            >
              Back to Sign In
            </button>
          </div>
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
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
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
