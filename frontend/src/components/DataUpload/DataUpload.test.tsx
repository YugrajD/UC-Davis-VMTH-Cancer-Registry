import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { IngestionJob } from '../../api/client';
import { parseCsvPreview } from './csvPreview';
import { DataUpload } from './DataUpload';

const mocks = vi.hoisted(() => ({
  authState: {
    user: null as { email?: string } | null,
    getAccessToken: vi.fn(),
  },
  uploadCSV: vi.fn(),
  fetchMyJobs: vi.fn(),
}));

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => mocks.authState,
}));

vi.mock('../../api/client', () => ({
  uploadCSV: mocks.uploadCSV,
  fetchMyJobs: mocks.fetchMyJobs,
  fetchFilterOptions: vi.fn().mockResolvedValue({ cancer_types: [], breeds: [], species: [] }),
}));

const completedJob: IngestionJob = {
  id: 1,
  uploaded_by_email: 'user@example.com',
  dataset_a_filename: 'clinical.csv',
  dataset_b_filename: 'demo.csv',
  status: 'completed',
  created_at: '2026-01-02T10:00:00.000Z',
};

const processingJob: IngestionJob = {
  ...completedJob,
  id: 2,
  dataset_a_filename: 'processing.csv',
  status: 'processing',
  processing_stage: 'reading_files',
};

function csvFile(name = 'clinical.csv') {
  const header = 'Date of Birth,Sex,Species,Breed,Zipcode Zipcode,RfrrVtrn Zipcode Zipcode,DtOfRq,Text';
  const row = '2020-01-01,Male,Canine,Labrador,95616,95616,2020-01-01,Normal pathology';
  return new File([`${header}\n${row}`], name, { type: 'text/csv' });
}

function invokeClickHandler(element: HTMLElement) {
  const propsKey = Object.keys(element).find(key => key.startsWith('__reactProps$'));
  const props = propsKey
    ? (element as unknown as Record<string, { onClick?: () => void } | undefined>)[propsKey]
    : undefined;
  props?.onClick?.();
}

beforeEach(() => {
  mocks.authState.user = null;
  mocks.authState.getAccessToken.mockResolvedValue(null);
  mocks.uploadCSV.mockResolvedValue(completedJob);
  mocks.fetchMyJobs.mockResolvedValue([]);
});

describe('DataUpload', () => {
  it('disables submit until a file and clinic name are provided', async () => {
    mocks.authState.user = { email: 'user@example.com' };
    const user = userEvent.setup();
    render(<DataUpload />);

    const submit = screen.getByRole('button', { name: /submit for review/i });
    expect(submit).toBeDisabled();

    await user.upload(screen.getByLabelText(/choose dataset file/i), csvFile());
    expect(submit).toBeDisabled(); // still disabled — no clinic name

    await user.type(screen.getByPlaceholderText(/uc davis vmth/i), 'VMTH');
    expect(submit).toBeEnabled();
  });

  it('shows a file required error if upload is invoked without a file', async () => {
    mocks.authState.user = { email: 'user@example.com' };
    render(<DataUpload />);

    const submit = screen.getByRole('button', { name: /submit for review/i });
    await act(async () => {
      invokeClickHandler(submit);
    });

    expect(await screen.findByText('Please select a dataset file.')).toBeInTheDocument();
  });

  it('shows the selected filename after choosing a file', async () => {
    mocks.authState.user = { email: 'user@example.com' };
    const user = userEvent.setup();
    render(<DataUpload />);

    await user.upload(screen.getByLabelText(/choose dataset file/i), csvFile('clinical.csv'));

    expect(screen.getByText('clinical.csv selected')).toBeInTheDocument();
  });

  it('uploads the file with token and clinic name, confirms submission, and refreshes jobs', async () => {
    const user = userEvent.setup();
    mocks.authState.user = { email: 'user@example.com' };
    mocks.authState.getAccessToken.mockResolvedValue('access-token');
    render(<DataUpload />);

    await waitFor(() => expect(mocks.fetchMyJobs).toHaveBeenCalledWith('access-token'));

    const file = csvFile('clinical.csv');
    await user.upload(screen.getByLabelText(/choose dataset file/i), file);
    await user.type(screen.getByPlaceholderText(/uc davis vmth/i), 'VMTH');
    await user.click(screen.getByRole('button', { name: /submit for review/i }));

    await waitFor(() => expect(mocks.uploadCSV).toHaveBeenCalledWith(file, 'access-token', 'VMTH'));
    expect(screen.getByText('Submitted for Review')).toBeInTheDocument();
    expect(mocks.fetchMyJobs).toHaveBeenCalledTimes(2);
  });

  it('shows upload failure messages', async () => {
    mocks.authState.user = { email: 'user@example.com' };
    const user = userEvent.setup();
    mocks.authState.getAccessToken.mockResolvedValue('access-token');
    mocks.uploadCSV.mockRejectedValue(new Error('Upload rejected'));
    render(<DataUpload />);

    await user.upload(screen.getByLabelText(/choose dataset file/i), csvFile());
    await user.type(screen.getByPlaceholderText(/uc davis vmth/i), 'VMTH');
    await user.click(screen.getByRole('button', { name: /submit for review/i }));

    expect(await screen.findByText('Upload rejected')).toBeInTheDocument();
  });

  it('reset clears files, submitted state, and errors', async () => {
    mocks.authState.user = { email: 'user@example.com' };
    const user = userEvent.setup();
    mocks.authState.getAccessToken.mockResolvedValue('access-token');
    render(<DataUpload />);

    await user.upload(screen.getByLabelText(/choose dataset file/i), csvFile('a.csv'));
    await user.type(screen.getByPlaceholderText(/uc davis vmth/i), 'VMTH');
    await user.click(screen.getByRole('button', { name: /submit for review/i }));
    expect(await screen.findByText('Submitted for Review')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /reset/i }));

    expect(screen.queryByText('a.csv selected')).not.toBeInTheDocument();
    expect(screen.queryByText('Submitted for Review')).not.toBeInTheDocument();
  });

  it('does not render My Uploads for signed-out users', () => {
    render(<DataUpload />);

    expect(screen.queryByText('My Uploads')).not.toBeInTheDocument();
    expect(mocks.fetchMyJobs).not.toHaveBeenCalled();
  });

  it('renders empty, loading, and populated upload states for signed-in users', async () => {
    mocks.authState.user = { email: 'user@example.com' };
    mocks.authState.getAccessToken.mockResolvedValue('token');
    let resolveJobs: (jobs: IngestionJob[]) => void = () => undefined;
    mocks.fetchMyJobs.mockReturnValueOnce(new Promise<IngestionJob[]>((resolve) => {
      resolveJobs = resolve;
    }));

    const { rerender, container } = render(<DataUpload />);
    expect(screen.getByText('My Uploads')).toBeInTheDocument();
    await waitFor(() => expect(container.querySelector('.animate-spin')).toBeInTheDocument());

    await act(async () => {
      resolveJobs([]);
    });
    expect(await screen.findByText('No uploads yet.')).toBeInTheDocument();

    mocks.fetchMyJobs.mockResolvedValueOnce([completedJob]);
    await userEvent.click(screen.getByRole('button', { name: /refresh/i }));

    expect(await screen.findByText('#1')).toBeInTheDocument();
    expect(screen.getByText('clinical.csv')).toBeInTheDocument();

    rerender(<DataUpload />);
    expect(screen.getByText('My Uploads')).toBeInTheDocument();
  });

  it('polls every 10 seconds while a job is processing and clears the interval on cleanup', async () => {
    mocks.authState.user = { email: 'user@example.com' };
    mocks.authState.getAccessToken.mockResolvedValue('token');
    mocks.fetchMyJobs.mockResolvedValue([processingJob]);
    const setIntervalSpy = vi.spyOn(globalThis, 'setInterval');
    const clearIntervalSpy = vi.spyOn(globalThis, 'clearInterval');

    const { unmount } = render(<DataUpload />);

    expect(await screen.findByText('processing.csv')).toBeInTheDocument();
    await waitFor(() => expect(setIntervalSpy).toHaveBeenCalledWith(expect.any(Function), 10000));

    unmount();

    expect(clearIntervalSpy).toHaveBeenCalled();
  });
});

describe('parseCsvPreview', () => {
  it('handles simple CSV, quoted commas, escaped quotes, blank cells, and empty files', () => {
    expect(parseCsvPreview('name,count\nLymphoma,2')).toEqual({
      headers: ['name', 'count'],
      rows: [['Lymphoma', '2']],
      totalRows: 1,
    });

    expect(parseCsvPreview('name,note\n"Smith, Ada","said ""hi""",\nblank,,')).toEqual({
      headers: ['name', 'note'],
      rows: [
        ['Smith, Ada', 'said "hi"', ''],
        ['blank', '', ''],
      ],
      totalRows: 2,
    });

    expect(parseCsvPreview('')).toEqual({ headers: [], rows: [], totalRows: 0 });
  });
});
