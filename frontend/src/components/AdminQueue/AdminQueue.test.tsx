import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi, type MockedFunction } from 'vitest';
import type { IngestionJob } from '../../api/client';
import { AdminQueue } from './AdminQueue';

const mocks = vi.hoisted(() => ({
  authState: {
    getAccessToken: vi.fn(),
  },
  fetchJobs: vi.fn(),
  reviewJob: vi.fn(),
  fetchJobPreview: vi.fn(),
  cancelJob: vi.fn(),
}));

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => mocks.authState,
}));

vi.mock('../../api/client', () => ({
  fetchJobs: mocks.fetchJobs,
  reviewJob: mocks.reviewJob,
  fetchJobPreview: mocks.fetchJobPreview,
  cancelJob: mocks.cancelJob,
}));

function confirmMock() {
  return globalThis.confirm as MockedFunction<typeof confirm>;
}

function alertMock() {
  return globalThis.alert as MockedFunction<typeof alert>;
}

function job(overrides: Partial<IngestionJob>): IngestionJob {
  return {
    id: 1,
    uploaded_by_email: 'uploader@example.com',
    dataset_a_filename: 'pending-a.csv',
    dataset_b_filename: 'pending-b.csv',
    status: 'pending_review',
    created_at: '2026-01-01T00:00:00.000Z',
    ...overrides,
  };
}

const jobs: IngestionJob[] = [
  job({ id: 1, status: 'pending_review', dataset_a_filename: 'pending-a.csv' }),
  job({ id: 2, status: 'processing', dataset_a_filename: 'local.csv', processing_stage: 'reading_files' }),
  job({ id: 3, status: 'completed', dataset_a_filename: 'completed.csv' }),
  job({ id: 4, status: 'failed', dataset_a_filename: 'failed.csv', processing_error: 'Pipeline failed' }),
  job({ id: 5, status: 'rejected', dataset_a_filename: 'rejected.csv', rejection_reason: 'Bad columns' }),
  job({ id: 6, status: 'cancelled', dataset_a_filename: 'cancelled.csv' }),
  job({ id: 7, status: 'processing', dataset_a_filename: 'gcp.csv', processing_stage: 'uploading_to_gcs' }),
];

beforeEach(() => {
  mocks.authState.getAccessToken.mockResolvedValue('admin-token');
  mocks.fetchJobs.mockResolvedValue(jobs);
  mocks.reviewJob.mockResolvedValue(job({ status: 'processing' }));
  mocks.fetchJobPreview.mockResolvedValue('col\nvalue');
  mocks.cancelJob.mockResolvedValue(job({ status: 'cancelled' }));
  confirmMock().mockReturnValue(true);
});

describe('AdminQueue', () => {
  it('loads jobs with a token and renders queue jobs separately from archive jobs', async () => {
    const user = userEvent.setup();
    render(<AdminQueue />);

    expect(await screen.findByText('Job #1')).toBeInTheDocument();
    expect(screen.getByText('Job #2')).toBeInTheDocument();
    expect(screen.getByText('Job #7')).toBeInTheDocument();
    expect(screen.queryByText('Job #3')).not.toBeInTheDocument();
    expect(mocks.fetchJobs).toHaveBeenCalledWith('admin-token');

    await user.click(screen.getByRole('button', { name: /archive\s*4/i }));

    expect(screen.getByText('Job #3')).toBeInTheDocument();
    expect(screen.getByText('Job #4')).toBeInTheDocument();
    expect(screen.getByText('Job #5')).toBeInTheDocument();
    expect(screen.getByText('Job #6')).toBeInTheDocument();
    expect(screen.queryByText('Job #1')).not.toBeInTheDocument();
  });

  it('exits loading without API calls when there is no token', async () => {
    mocks.authState.getAccessToken.mockResolvedValue(null);
    render(<AdminQueue />);

    expect(await screen.findByText('No active jobs in the queue.')).toBeInTheDocument();
    expect(mocks.fetchJobs).not.toHaveBeenCalled();
  });

  it('displays API load failures', async () => {
    mocks.fetchJobs.mockRejectedValue(new Error('Queue unavailable'));
    render(<AdminQueue />);

    expect(await screen.findByText('Queue unavailable')).toBeInTheDocument();
  });

  it('counts active and archived jobs by status', async () => {
    render(<AdminQueue />);

    expect(await screen.findByRole('button', { name: /queue\s*3/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /archive\s*4/i })).toBeInTheDocument();
  });

  it('filters the archive by selected status', async () => {
    const user = userEvent.setup();
    render(<AdminQueue />);

    await user.click(await screen.findByRole('button', { name: /archive/i }));
    await user.selectOptions(screen.getByRole('combobox'), 'failed');

    expect(screen.getByText('Job #4')).toBeInTheDocument();
    expect(screen.queryByText('Job #3')).not.toBeInTheDocument();
    expect(screen.queryByText('Job #5')).not.toBeInTheDocument();
  });

  it('does not approve when confirmation is cancelled', async () => {
    const user = userEvent.setup();
    confirmMock().mockReturnValueOnce(false);
    render(<AdminQueue />);

    await screen.findByText('Job #1');
    await user.click(screen.getByRole('button', { name: /approve/i }));

    expect(mocks.reviewJob).not.toHaveBeenCalled();
  });

  it('approves jobs and reloads the queue', async () => {
    const user = userEvent.setup();
    render(<AdminQueue />);

    await screen.findByText('Job #1');
    await user.click(screen.getByRole('button', { name: /approve/i }));

    await waitFor(() => expect(mocks.reviewJob).toHaveBeenCalledWith('admin-token', 1, 'approve'));
    expect(mocks.fetchJobs).toHaveBeenCalledTimes(2);
  });

  it('rejects jobs with an optional reason, clears reject state, and reloads', async () => {
    const user = userEvent.setup();
    render(<AdminQueue />);

    await screen.findByText('Job #1');
    await user.click(screen.getByRole('button', { name: /^reject$/i }));
    await user.type(screen.getByPlaceholderText(/rejection reason/i), 'Missing required columns');
    await user.click(screen.getByRole('button', { name: /confirm reject/i }));

    await waitFor(() => {
      expect(mocks.reviewJob).toHaveBeenCalledWith('admin-token', 1, 'reject', 'Missing required columns');
    });
    expect(screen.queryByPlaceholderText(/rejection reason/i)).not.toBeInTheDocument();
    expect(mocks.fetchJobs).toHaveBeenCalledTimes(2);
  });

  it('cancels processing jobs after confirmation and reloads', async () => {
    const user = userEvent.setup();
    render(<AdminQueue />);

    await screen.findByText('Job #2');
    await user.click(screen.getAllByRole('button', { name: /cancel/i })[0]);

    await waitFor(() => expect(mocks.cancelJob).toHaveBeenCalledWith('admin-token', 2));
    expect(mocks.fetchJobs).toHaveBeenCalledTimes(2);
  });

  it('respects cancelled cancel confirmation', async () => {
    const user = userEvent.setup();
    confirmMock().mockReturnValueOnce(false);
    render(<AdminQueue />);

    await screen.findByText('Job #2');
    await user.click(screen.getAllByRole('button', { name: /cancel/i })[0]);

    expect(mocks.cancelJob).not.toHaveBeenCalled();
  });

  it('loads previews, disables the active preview button, and closes on Escape', async () => {
    const user = userEvent.setup();
    let resolvePreview: (content: string) => void = () => undefined;
    mocks.fetchJobPreview.mockReturnValueOnce(new Promise<string>((resolve) => {
      resolvePreview = resolve;
    }));
    render(<AdminQueue />);

    await screen.findByText('Job #1');
    const previewButton = screen.getAllByRole('button', { name: /preview a/i })[0];
    await user.click(previewButton);

    expect(previewButton).toBeDisabled();
    expect(mocks.fetchJobPreview).toHaveBeenCalledWith('admin-token', 1, 'a');

    resolvePreview('col\nvalue123');
    expect(await screen.findByText('value123')).toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'Escape' });

    await waitFor(() => expect(screen.queryByText('value123')).not.toBeInTheDocument());
  });

  it('alerts action failures', async () => {
    const user = userEvent.setup();
    mocks.reviewJob.mockRejectedValueOnce(new Error('Approve failed upstream'));
    render(<AdminQueue />);

    await screen.findByText('Job #1');
    await user.click(screen.getByRole('button', { name: /approve/i }));

    await waitFor(() => expect(alertMock()).toHaveBeenCalledWith('Approve failed upstream'));
  });

  it('renders local and GCP processing stage indicators', async () => {
    render(<AdminQueue />);

    expect(await screen.findByText('Reading files')).toBeInTheDocument();
    expect(screen.getByText('Uploading to Cloud Storage')).toBeInTheDocument();
    expect(screen.getByText('Submitting batch job')).toBeInTheDocument();
  });
});
