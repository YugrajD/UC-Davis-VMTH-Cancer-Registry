import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { User } from '@supabase/supabase-js';
import { Settings } from './Settings';
import { HIDE_PREDICTIONS_BY_DEFAULT_KEY } from '../../lib/auditWorklistPrefs';

const mockAuthState = vi.hoisted(() => ({
  user: { email: 'reviewer@ucdavis.edu' } as User,
  isAdmin: false,
  isUploader: false,
  isReviewer: false,
}));

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => mockAuthState,
}));

describe('Settings', () => {
  it('hides the Audit Worklist preferences section for a non-admin user', () => {
    mockAuthState.isReviewer = false;
    mockAuthState.isAdmin = false;
    render(<Settings />);

    expect(screen.queryByText('Hide model predictions by default')).not.toBeInTheDocument();
  });

  it('hides the preference from a reviewer who is not an admin', () => {
    mockAuthState.isReviewer = true;
    mockAuthState.isAdmin = false;
    render(<Settings />);

    expect(screen.queryByText('Hide model predictions by default')).not.toBeInTheDocument();
  });

  it('shows the preference for an admin and persists it to localStorage', async () => {
    mockAuthState.isReviewer = false;
    mockAuthState.isAdmin = true;
    const user = userEvent.setup();
    render(<Settings />);

    const checkbox = screen.getByRole('checkbox', { name: /Hide model predictions by default/ });
    expect(checkbox).not.toBeChecked();

    await user.click(checkbox);

    expect(checkbox).toBeChecked();
    expect(localStorage.getItem(HIDE_PREDICTIONS_BY_DEFAULT_KEY)).toBe('true');
  });
});
