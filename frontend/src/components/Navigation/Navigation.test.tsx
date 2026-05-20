import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Navigation } from './Navigation';

const mocks = vi.hoisted(() => ({
  authState: {
    user: null as { email?: string } | null,
    isAdmin: false,
    loading: false,
    signOut: vi.fn(),
    signIn: vi.fn(),
  },
}));

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => mocks.authState,
}));

beforeEach(() => {
  mocks.authState.user = null;
  mocks.authState.isAdmin = false;
  mocks.authState.loading = false;
});

describe('Navigation', () => {
  it('hides Review Queue for non-admins and shows it for admins', () => {
    const { rerender } = render(<Navigation activeTab="overview" onTabChange={vi.fn()} />);

    expect(screen.queryByRole('button', { name: /review queue/i })).not.toBeInTheDocument();

    mocks.authState.isAdmin = true;
    rerender(<Navigation activeTab="overview" onTabChange={vi.fn()} />);

    expect(screen.getByRole('button', { name: /review queue/i })).toBeInTheDocument();
  });

  it('shows Sign In for signed-out users and opens the login modal', async () => {
    const user = userEvent.setup();
    render(<Navigation activeTab="overview" onTabChange={vi.fn()} />);

    await user.click(screen.getByRole('button', { name: /sign in/i }));

    expect(screen.getByRole('heading', { name: /sign in/i })).toBeInTheDocument();
  });

  it('shows email, Admin badge, and Sign Out for signed-in admins', async () => {
    const user = userEvent.setup();
    mocks.authState.user = { email: 'admin@example.com' };
    mocks.authState.isAdmin = true;
    render(<Navigation activeTab="overview" onTabChange={vi.fn()} />);

    expect(screen.getByText('admin@example.com')).toBeInTheDocument();
    expect(screen.getByText('Admin')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /sign out/i }));
    expect(mocks.authState.signOut).toHaveBeenCalled();
  });
});
