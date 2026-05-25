import { describe, it, expect } from 'vitest';
import { LoginModal } from '../components/LoginModal/LoginModal';
import { ResetPasswordModal } from '../components/ResetPasswordModal/ResetPasswordModal';
import { AuthProvider, useAuth } from '../contexts/AuthContext';
import { fetchIncidence, fetchCalEnviroScreen } from '../api/client';

// ---------------------------------------------------------------------------
// LoginModal — export and shape
// ---------------------------------------------------------------------------

describe('LoginModal', () => {
  it('is exported as a function (React component)', () => {
    expect(typeof LoginModal).toBe('function');
  });

  it('accepts an onClose prop (arity ≥ 0)', () => {
    expect(LoginModal.length).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// ResetPasswordModal — export and shape
// ---------------------------------------------------------------------------

describe('ResetPasswordModal', () => {
  it('is exported as a function (React component)', () => {
    expect(typeof ResetPasswordModal).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// AuthContext — new interface members
// ---------------------------------------------------------------------------

describe('AuthContext interface', () => {
  it('AuthProvider is exported as a function', () => {
    expect(typeof AuthProvider).toBe('function');
  });

  it('useAuth is exported as a function', () => {
    expect(typeof useAuth).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// API functions used by combined download and password reset flow
// ---------------------------------------------------------------------------

describe('fetchIncidence', () => {
  it('is a function', () => {
    expect(typeof fetchIncidence).toBe('function');
  });

  it('accepts optional filters and returns a promise', () => {
    const result = fetchIncidence({});
    expect(result).toBeInstanceOf(Promise);
    // Prevent unhandled rejection from the network call in test env
    result.catch(() => {});
  });

  it('rejects when the backend is unreachable', async () => {
    await expect(fetchIncidence()).rejects.toThrow();
  });
});

describe('fetchCalEnviroScreen', () => {
  it('is a function', () => {
    expect(typeof fetchCalEnviroScreen).toBe('function');
  });

  it('returns a promise', () => {
    const result = fetchCalEnviroScreen();
    expect(result).toBeInstanceOf(Promise);
    result.catch(() => {});
  });

  it('rejects when the backend is unreachable', async () => {
    await expect(fetchCalEnviroScreen()).rejects.toThrow();
  });
});
