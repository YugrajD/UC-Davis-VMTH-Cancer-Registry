import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

async function loadLoginModal({
  configured = true,
  signIn = vi.fn().mockResolvedValue(undefined),
}: {
  configured?: boolean;
  signIn?: ReturnType<typeof vi.fn>;
} = {}) {
  vi.resetModules();
  vi.doMock('../../contexts/AuthContext', () => ({
    useAuth: () => ({
      signIn,
      signUp: vi.fn(),
      confirmSignUp: vi.fn(),
      resendConfirmationCode: vi.fn(),
      forgotPassword: vi.fn(),
      confirmForgotPassword: vi.fn(),
      signInWithGoogle: vi.fn(),
      googleOAuthConfigured: false,
      clearAuthError: vi.fn(),
      authError: null,
    }),
  }));
  vi.doMock('../../lib/cognito', () => ({
    authConfigured: configured,
  }));

  const module = await import('./LoginModal');
  return { LoginModal: module.LoginModal, signIn };
}

afterEach(() => {
  vi.resetModules();
  vi.doUnmock('../../contexts/AuthContext');
  vi.doUnmock('../../lib/cognito');
});

describe('LoginModal', () => {
  it('renders an auth configuration warning when Cognito is unconfigured', async () => {
    const { LoginModal } = await loadLoginModal({ configured: false });

    render(<LoginModal onClose={vi.fn()} />);

    expect(screen.getByText('Auth not configured')).toBeInTheDocument();
    expect(screen.getByText('VITE_COGNITO_USER_POOL_ID')).toBeInTheDocument();
    expect(screen.getByText('VITE_COGNITO_CLIENT_ID')).toBeInTheDocument();
  });

  it('submits credentials and closes on successful login', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const signIn = vi.fn().mockResolvedValue(undefined);
    const { LoginModal } = await loadLoginModal({ signIn });
    render(<LoginModal onClose={onClose} />);

    await user.type(screen.getByPlaceholderText(/you@example.com/i), 'admin@example.com');
    await user.type(screen.getByPlaceholderText(/password/i), 'secret');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => expect(signIn).toHaveBeenCalledWith('admin@example.com', 'secret'));
    expect(onClose).toHaveBeenCalled();
  });

  it('displays failed login errors', async () => {
    const user = userEvent.setup();
    const signIn = vi.fn().mockRejectedValue(new Error('Invalid login'));
    const { LoginModal } = await loadLoginModal({ signIn });
    render(<LoginModal onClose={vi.fn()} />);

    await user.type(screen.getByPlaceholderText(/you@example.com/i), 'admin@example.com');
    await user.type(screen.getByPlaceholderText(/password/i), 'wrong');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    expect(await screen.findByText('Invalid login')).toBeInTheDocument();
  });

  it('closes on Escape and backdrop click', async () => {
    const { LoginModal } = await loadLoginModal();
    const onClose = vi.fn();
    const { container } = render(<LoginModal onClose={onClose} />);

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);

    const backdropTarget = container.firstElementChild;
    if (!backdropTarget) throw new Error('Backdrop target missing');
    fireEvent.click(backdropTarget);

    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it('moves to the confirmation-code step after signing up', async () => {
    const user = userEvent.setup();
    const { LoginModal } = await loadLoginModal();
    render(<LoginModal onClose={vi.fn()} />);

    await user.click(screen.getByRole('button', { name: /sign up/i }));
    await user.type(screen.getByPlaceholderText(/you@example.com/i), 'new@example.com');
    await user.type(screen.getByPlaceholderText('Password'), 'Password123!');
    await user.type(screen.getByPlaceholderText(/confirm password/i), 'Password123!');
    await user.click(screen.getByRole('button', { name: /create account/i }));

    expect(await screen.findByPlaceholderText('123456')).toBeInTheDocument();
  });
});
