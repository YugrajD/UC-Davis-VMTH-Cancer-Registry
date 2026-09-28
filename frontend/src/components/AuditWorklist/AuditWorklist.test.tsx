import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AuditCaseDetail, TaxonomyTermOut, WorklistResponse } from '../../api/client';
import { AuditWorklist } from './AuditWorklist';

const mocks = vi.hoisted(() => ({
  authState: {
    getAccessToken: vi.fn(),
    isAdmin: false,
  },
  fetchAuditWorklist: vi.fn(),
  fetchTaxonomyTerms: vi.fn(),
  fetchAuditCaseDetail: vi.fn(),
  saveCaseReview: vi.fn(),
  reopenCaseReview: vi.fn(),
  importAuditList: vi.fn(),
  createGoldExport: vi.fn(),
  downloadGoldExport: vi.fn(),
}));

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => mocks.authState,
}));

vi.mock('../../api/client', () => ({
  fetchAuditWorklist: mocks.fetchAuditWorklist,
  fetchTaxonomyTerms: mocks.fetchTaxonomyTerms,
  fetchAuditCaseDetail: mocks.fetchAuditCaseDetail,
  saveCaseReview: mocks.saveCaseReview,
  reopenCaseReview: mocks.reopenCaseReview,
  importAuditList: mocks.importAuditList,
  createGoldExport: mocks.createGoldExport,
  downloadGoldExport: mocks.downloadGoldExport,
}));

const worklist: WorklistResponse = {
  list_id: '2026-09-27-2',
  imported_at: '2026-09-27T00:00:00Z',
  case_count: 2,
  cases: [
    {
      case_id: 'CASE-0001',
      position: 0,
      patient_found: true,
      review_status: 'unreviewed',
      no_cancer: null,
      code_count: 0,
      reviewed_by_email: null,
      reviewed_at: null,
    },
    {
      case_id: 'CASE-9999',
      position: 1,
      patient_found: false,
      review_status: 'locked',
      no_cancer: true,
      code_count: 0,
      reviewed_by_email: 'dr.smith@ucdavis.edu',
      reviewed_at: '2026-09-27T00:00:00Z',
    },
  ],
};

const terms: TaxonomyTermOut[] = [
  { vet_icd_o_code: '9590/3', taxonomy_group: 'Malignant lymphomas', taxonomy_term: 'Malignant lymphoma, NOS' },
];

function caseDetail(overrides: Partial<AuditCaseDetail> = {}): AuditCaseDetail {
  return {
    case_id: 'CASE-0001',
    patient_found: true,
    patient_anon_id: 'CASE-0001',
    source_diagnosis: 'Skin mass',
    report_text: 'Full pathology report text.',
    predicted_codes: [
      {
        diagnosis_index: 1,
        cancer_type_name: 'Mast cell neoplasms',
        icd_o_code: '9740/3',
        predicted_term: 'Mast cell tumor, NOS',
        confidence: 0.91,
        prediction_method: 'embedding',
      },
    ],
    review_exists: false,
    review_no_cancer: null,
    review_codes: [],
    review_locked: false,
    reviewed_by_email: null,
    reviewed_at: null,
    ...overrides,
  };
}

beforeEach(() => {
  mocks.authState.getAccessToken.mockResolvedValue('reviewer-token');
  mocks.authState.isAdmin = false;
  mocks.fetchAuditWorklist.mockResolvedValue(worklist);
  mocks.fetchTaxonomyTerms.mockResolvedValue(terms);
  mocks.fetchAuditCaseDetail.mockResolvedValue(caseDetail());
  mocks.saveCaseReview.mockResolvedValue({
    case_id: 'CASE-0001', no_cancer: false, code_count: 1,
    reviewed_by_email: 'reviewer@ucdavis.edu', reviewed_at: '2026-09-27T00:00:00Z', locked: false,
  });
});

describe('AuditWorklist', () => {
  it('loads the worklist and shows each case with its status', async () => {
    render(<AuditWorklist />);

    expect(await screen.findByText('CASE-0001')).toBeInTheDocument();
    expect(screen.getByText('CASE-9999')).toBeInTheDocument();
    expect(screen.getByText('Unreviewed')).toBeInTheDocument();
    expect(screen.getByText('Exported')).toBeInTheDocument();
    expect(screen.getByText('No record')).toBeInTheDocument();
    expect(mocks.fetchAuditWorklist).toHaveBeenCalledWith('reviewer-token');
  });

  it('shows an empty-worklist message when there is no active list', async () => {
    mocks.fetchAuditWorklist.mockResolvedValue({ list_id: null, imported_at: null, case_count: 0, cases: [] });
    render(<AuditWorklist />);

    expect(await screen.findByText('No active audit list.')).toBeInTheDocument();
  });

  it('loads a case detail on selection and shows its report and predictions', async () => {
    const user = userEvent.setup();
    render(<AuditWorklist />);

    await user.click(await screen.findByText('CASE-0001'));

    expect(await screen.findByText('Full pathology report text.')).toBeInTheDocument();
    expect(screen.getByText('Skin mass')).toBeInTheDocument();
    expect(screen.getByText(/Mast cell tumor, NOS/)).toBeInTheDocument();
    expect(mocks.fetchAuditCaseDetail).toHaveBeenCalledWith('reviewer-token', 'CASE-0001');
  });

  it('adds a taxonomy code via the picker and saves the review', async () => {
    const user = userEvent.setup();
    render(<AuditWorklist />);

    await user.click(await screen.findByText('CASE-0001'));
    await screen.findByText('Full pathology report text.');

    const search = screen.getByPlaceholderText('Search taxonomy terms…');
    await user.click(search);
    await user.click(await screen.findByRole('option'));

    expect(screen.getByText('Malignant lymphoma, NOS')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Save review' }));

    await waitFor(() => {
      expect(mocks.saveCaseReview).toHaveBeenCalledWith('reviewer-token', 'CASE-0001', {
        no_cancer: false,
        codes: [{ taxonomy_group: 'Malignant lymphomas', taxonomy_term: 'Malignant lymphoma, NOS' }],
      });
    });
    expect(await screen.findByText('Saved.')).toBeInTheDocument();
  });

  it('disables the review form and offers Reopen (admin only) when locked', async () => {
    mocks.authState.isAdmin = true;
    mocks.fetchAuditCaseDetail.mockResolvedValue(
      caseDetail({ review_locked: true, review_exists: true, reviewed_by_email: 'dr.smith@ucdavis.edu' }),
    );
    const user = userEvent.setup();
    render(<AuditWorklist />);

    await user.click(await screen.findByText('CASE-0001'));

    expect(await screen.findByText(/Locked — exported by dr.smith@ucdavis.edu/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save review' })).toBeDisabled();

    mocks.reopenCaseReview.mockResolvedValue({ case_id: 'CASE-0001', locked: false });
    await user.click(screen.getByRole('button', { name: 'Reopen' }));

    await waitFor(() => {
      expect(mocks.reopenCaseReview).toHaveBeenCalledWith('reviewer-token', 'CASE-0001');
    });
  });

  it('imports an audit list from the admin tools panel', async () => {
    mocks.authState.isAdmin = true;
    mocks.importAuditList.mockResolvedValue({
      list_id: '2026-09-28-1', case_count: 5, replaced_list_id: '2026-09-27-2', not_found: [],
    });
    const user = userEvent.setup();
    render(<AuditWorklist />);

    await user.click(await screen.findByText('+ Admin tools'));

    const listInput = screen.getByLabelText('List (.txt)') as HTMLInputElement;
    const manifestInput = screen.getByLabelText('Manifest (.manifest.json)') as HTMLInputElement;
    const listFile = new File(['CASE-0001\n'], 'audit_list_2026-09-28-1.txt', { type: 'text/plain' });
    const manifestFile = new File(['{}'], 'audit_list_2026-09-28-1.txt.manifest.json', { type: 'application/json' });
    await user.upload(listInput, listFile);
    await user.upload(manifestInput, manifestFile);

    await user.click(screen.getByRole('button', { name: 'Import' }));

    await waitFor(() => {
      expect(mocks.importAuditList).toHaveBeenCalledWith('reviewer-token', listFile, manifestFile);
    });
    expect(await screen.findByText(/Imported 2026-09-28-1/)).toBeInTheDocument();
  });
});
