import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AuditCaseDetail, TaxonomyTermOut, WorklistResponse } from '../../api/client';
import { AuditWorklist } from './AuditWorklist';
import realTaxonomy from './__fixtures__/taxonomyTerms.fixture.json';

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
  {
    vet_icd_o_code: '9590/3',
    taxonomy_group: 'Malignant lymphomas',
    taxonomy_term: 'Malignant lymphoma, NOS',
    term_level: 'Preferred',
  },
];

function caseDetail(overrides: Partial<AuditCaseDetail> = {}): AuditCaseDetail {
  return {
    case_id: 'CASE-0001',
    patient_found: true,
    patient_anon_id: 'CASE-0001',
    patient_species: 'Dog',
    patient_breed: 'Labrador Retriever',
    patient_sex: 'Spayed Female',
    patient_age: 7,
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
        code_source: 'report',
        source_confidence: '0.91',
        ml_review_status: 'auto_accepted',
      },
    ],
    registry_no_cancer: false,
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

  it('paginates the worklist 10 cases at a time', async () => {
    const user = userEvent.setup();
    const manyCases: WorklistResponse = {
      list_id: 'big-list',
      imported_at: '2026-09-27T00:00:00Z',
      case_count: 25,
      cases: Array.from({ length: 25 }, (_, i) => ({
        case_id: `CASE-${String(i).padStart(4, '0')}`,
        position: i,
        patient_found: true,
        review_status: 'unreviewed' as const,
        no_cancer: null,
        code_count: 0,
        reviewed_by_email: null,
        reviewed_at: null,
      })),
    };
    mocks.fetchAuditWorklist.mockResolvedValue(manyCases);
    render(<AuditWorklist />);

    // Page 1: first 10 cases only, no page-11-only case visible.
    expect(await screen.findByText('CASE-0000')).toBeInTheDocument();
    expect(screen.getByText('CASE-0009')).toBeInTheDocument();
    expect(screen.queryByText('CASE-0010')).not.toBeInTheDocument();
    expect(screen.getByText('Page 1 of 3')).toBeInTheDocument();
    expect(screen.getByText('Previous')).toBeDisabled();

    await user.click(screen.getByText('Next'));
    expect(await screen.findByText('CASE-0010')).toBeInTheDocument();
    expect(screen.getByText('CASE-0019')).toBeInTheDocument();
    expect(screen.queryByText('CASE-0000')).not.toBeInTheDocument();
    expect(screen.getByText('Page 2 of 3')).toBeInTheDocument();

    await user.click(screen.getByText('Next'));
    expect(await screen.findByText('CASE-0020')).toBeInTheDocument();
    expect(screen.getByText('CASE-0024')).toBeInTheDocument();
    expect(screen.getByText('Page 3 of 3')).toBeInTheDocument();
    expect(screen.getByText('Next')).toBeDisabled();

    await user.click(screen.getByText('Previous'));
    expect(await screen.findByText('CASE-0010')).toBeInTheDocument();
    expect(screen.getByText('Page 2 of 3')).toBeInTheDocument();
  });

  it('loads a case detail on selection and shows its report, demographics, and predictions', async () => {
    const user = userEvent.setup();
    render(<AuditWorklist />);

    await user.click(await screen.findByText('CASE-0001'));

    expect(await screen.findByText('Full pathology report text.')).toBeInTheDocument();
    expect(screen.getByText('Skin mass')).toBeInTheDocument();
    expect(screen.getByText('Dog · Labrador Retriever · Spayed Female · 7 yrs')).toBeInTheDocument();
    // Appears twice: once in the read-only "Current predictions" list, once
    // pre-filled into the editable code set (approve-or-correct).
    expect(screen.getAllByText(/Mast cell tumor, NOS/)).toHaveLength(2);
    expect(mocks.fetchAuditCaseDetail).toHaveBeenCalledWith('reviewer-token', 'CASE-0001');
  });

  it('singularizes "1 yr" and omits age entirely when birth_date was unavailable', async () => {
    const user = userEvent.setup();
    mocks.fetchAuditCaseDetail.mockResolvedValueOnce(caseDetail({ patient_age: 1 }));
    render(<AuditWorklist />);

    await user.click(await screen.findByText('CASE-0001'));
    expect(await screen.findByText('Dog · Labrador Retriever · Spayed Female · 1 yr')).toBeInTheDocument();

    cleanup();
    mocks.fetchAuditCaseDetail.mockResolvedValueOnce(caseDetail({ patient_age: null }));
    render(<AuditWorklist />);

    await user.click(await screen.findByText('CASE-0001'));
    expect(await screen.findByText('Dog · Labrador Retriever · Spayed Female')).toBeInTheDocument();
  });

  it('pre-fills from the registry prediction, offering Approve, and lets a correction be saved', async () => {
    const user = userEvent.setup();
    render(<AuditWorklist />);

    await user.click(await screen.findByText('CASE-0001'));
    await screen.findByText('Full pathology report text.');

    // Unmodified from the pre-filled prediction: the button reads Approve.
    expect(screen.getByRole('button', { name: 'Approve' })).toBeInTheDocument();

    const termBox = screen.getByRole('combobox', { name: 'Term' });
    await user.type(termBox, 'Malignant lymphoma');
    await user.click(await screen.findByRole('option'));

    // The code box fills in from the picked term, and the pair isn't added
    // to the case until Add is clicked.
    expect(screen.getByRole('combobox', { name: 'ICD-O code' })).toHaveValue('9590/3');
    await user.click(screen.getByRole('button', { name: 'Add' }));

    expect(screen.getByText('Malignant lymphoma, NOS')).toBeInTheDocument();

    // Adding a code changes it from an approval to a correction.
    await user.click(screen.getByRole('button', { name: 'Save correction' }));

    await waitFor(() => {
      expect(mocks.saveCaseReview).toHaveBeenCalledWith('reviewer-token', 'CASE-0001', {
        no_cancer: false,
        codes: [
          { taxonomy_group: 'Mast cell neoplasms', taxonomy_term: 'Mast cell tumor, NOS' },
          { taxonomy_group: 'Malignant lymphomas', taxonomy_term: 'Malignant lymphoma, NOS' },
        ],
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
    expect(screen.getByRole('button', { name: 'Approve' })).toBeDisabled();

    mocks.reopenCaseReview.mockResolvedValue({ case_id: 'CASE-0001', locked: false });
    await user.click(screen.getByRole('button', { name: 'Reopen' }));

    await waitFor(() => {
      expect(mocks.reopenCaseReview).toHaveBeenCalledWith('reviewer-token', 'CASE-0001');
    });
  });

  it('pre-fills the no-cancer checkbox and lets it be approved when the registry already coded the case cancer-free', async () => {
    mocks.fetchAuditCaseDetail.mockResolvedValue(
      caseDetail({ predicted_codes: [], registry_no_cancer: true }),
    );
    const user = userEvent.setup();
    render(<AuditWorklist />);

    await user.click(await screen.findByText('CASE-0001'));
    await screen.findByText('Registry: no reportable cancer.');

    const checkbox = screen.getByRole('checkbox', { name: 'No reportable cancer' }) as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
    expect(screen.getByRole('button', { name: 'Approve' })).toBeEnabled();

    await user.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() => {
      expect(mocks.saveCaseReview).toHaveBeenCalledWith('reviewer-token', 'CASE-0001', {
        no_cancer: true,
        codes: [],
      });
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

  // Code picker acceptance checks, run against the real Vet-ICD-O-canine-1
  // taxonomy (845 rows, parsed from ml/taxonomy/labels.csv) rather than the
  // small fixture above, so the literal codes/terms/counts asserted below
  // are the picker's actual behavior on production data.
  describe('Code picker', () => {
    beforeEach(() => {
      mocks.fetchTaxonomyTerms.mockResolvedValue(realTaxonomy as TaxonomyTermOut[]);
    });

    async function openCase() {
      const user = userEvent.setup();
      render(<AuditWorklist />);
      await user.click(await screen.findByText('CASE-0001'));
      await screen.findByText('Full pathology report text.');
      return user;
    }

    async function pickOption(user: ReturnType<typeof userEvent.setup>, pattern: RegExp) {
      const option = await waitFor(() => {
        const match = screen.getAllByRole('option').find((el) => pattern.test(el.textContent ?? ''));
        expect(match).toBeTruthy();
        return match as HTMLElement;
      });
      await user.click(option);
    }

    it('1. resolves a code to its Preferred term and group, enabling Add', async () => {
      const user = await openCase();
      const codeBox = screen.getByRole('combobox', { name: 'ICD-O code' });
      await user.type(codeBox, '8010/3');
      await pickOption(user, /Carcinoma, NOS/);

      expect(screen.getByRole('combobox', { name: 'Term' })).toHaveValue('Carcinoma, NOS');
      expect(screen.getByText(/Epithelial neoplasms, NOS/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Add' })).toBeEnabled();
    });

    it('2. accepts 80103, "8010 3" and "8010-3" as the same code, listed first', async () => {
      for (const digits of ['80103', '8010 3', '8010-3']) {
        const user = await openCase();
        const codeBox = screen.getByRole('combobox', { name: 'ICD-O code' });
        await user.type(codeBox, digits);
        const options = await screen.findAllByRole('option');
        expect(options[0]).toHaveTextContent('8010/3');
        cleanup();
      }
    });

    it('3. focusing Term after a code-driven pair shows its synonyms, keeping the code', async () => {
      const user = await openCase();
      const codeBox = screen.getByRole('combobox', { name: 'ICD-O code' });
      const termBox = screen.getByRole('combobox', { name: 'Term' });
      await user.type(codeBox, '8010/3');
      await pickOption(user, /Carcinoma, NOS/);

      await user.click(termBox);
      await pickOption(user, /Epithelial tumor, malignant/);

      expect(codeBox).toHaveValue('8010/3');
      expect(termBox).toHaveValue('Epithelial tumor, malignant');
    });

    it('4. tolerates a typo in the term and resolves to the right code', async () => {
      const user = await openCase();
      const termBox = screen.getByRole('combobox', { name: 'Term' });
      await user.type(termBox, 'hemangiosarcma');

      const option = await waitFor(() => {
        const match = screen.getAllByRole('option').find((el) => /Hemangiosarcoma, NOS/.test(el.textContent ?? ''));
        expect(match).toBeTruthy();
        return match as HTMLElement;
      });
      expect(option).toHaveTextContent('close match');
      await userEvent.setup().click(option);

      expect(screen.getByRole('combobox', { name: 'ICD-O code' })).toHaveValue('9120/3');
    });

    it('5. caps results at 25 with a "Showing 25 of 82" footer for a broad term query', async () => {
      const user = await openCase();
      const termBox = screen.getByRole('combobox', { name: 'Term' });
      await user.type(termBox, 'lymphoma');

      expect(await screen.findAllByRole('option')).toHaveLength(25);
      expect(screen.getByText('Showing 25 of 82. Keep typing to narrow.')).toBeInTheDocument();
    });

    it('6. flags a term listed under two codes as ambiguous on blur', async () => {
      const user = await openCase();
      const termBox = screen.getByRole('combobox', { name: 'Term' });
      await user.type(termBox, 'Papillary adenocarcinoma');
      await user.tab();

      expect(await screen.findByRole('alert')).toHaveTextContent(
        'Papillary adenocarcinoma is listed under 2 codes (8050/3, 8260/3). Pick one from the list.',
      );
      expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled();
    });

    it('7. flags an unknown code on blur', async () => {
      const user = await openCase();
      const codeBox = screen.getByRole('combobox', { name: 'ICD-O code' });
      await user.type(codeBox, '8011/3');
      await user.tab();

      expect(await screen.findByRole('alert')).toHaveTextContent(
        'No code matches ‘8011/3’. Pick one from the list.',
      );
      expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled();
    });

    it('8. editing the code after a pair clears the term until a new code is picked', async () => {
      const user = await openCase();
      const codeBox = screen.getByRole('combobox', { name: 'ICD-O code' });
      const termBox = screen.getByRole('combobox', { name: 'Term' });
      await user.type(codeBox, '8010/3');
      await pickOption(user, /Carcinoma, NOS/);

      await user.clear(codeBox);
      await user.type(codeBox, '9120');
      expect(termBox).toHaveValue('');

      await pickOption(user, /^9120\/3.*Hemangiosarcoma, NOS/s);
      expect(termBox).toHaveValue('Hemangiosarcoma, NOS');
    });

    it('9. Esc while editing the code restores the previous complete pair', async () => {
      const user = await openCase();
      const codeBox = screen.getByRole('combobox', { name: 'ICD-O code' });
      const termBox = screen.getByRole('combobox', { name: 'Term' });
      await user.type(codeBox, '8010/3');
      await pickOption(user, /Carcinoma, NOS/);

      await user.type(codeBox, '9');
      expect(codeBox).toHaveValue('8010/39');

      await user.keyboard('{Escape}');

      expect(codeBox).toHaveValue('8010/3');
      expect(termBox).toHaveValue('Carcinoma, NOS');
    });

    it('10. adding the same pair twice shows "Already added" without duplicating the chip', async () => {
      const user = await openCase();
      const codeBox = screen.getByRole('combobox', { name: 'ICD-O code' });
      await user.type(codeBox, '8010/3');
      await pickOption(user, /Carcinoma, NOS/);
      await user.click(screen.getByRole('button', { name: 'Add' }));

      await user.type(codeBox, '8010/3');
      await pickOption(user, /Carcinoma, NOS/);
      await user.click(screen.getByRole('button', { name: 'Add' }));

      expect(screen.getByText(/Already added: 8010\/3 Carcinoma, NOS/)).toBeInTheDocument();
      // Only one chip for the pair — plus the case's pre-filled Mast cell
      // prediction, never duplicated.
      expect(screen.getAllByText('Carcinoma, NOS')).toHaveLength(1);
      expect(screen.getAllByRole('button', { name: 'Remove' })).toHaveLength(2);
    });

    it('shows a hint instead of code results when letters are typed into the Code box', async () => {
      const user = await openCase();
      const codeBox = screen.getByRole('combobox', { name: 'ICD-O code' });
      await user.type(codeBox, 'carc');

      expect(
        await screen.findByText('Codes are numbers like 8010/3. To search by name, use the Term box.'),
      ).toBeInTheDocument();
    });

    // 11. Saving still sends {taxonomy_group, taxonomy_term} per code — see
    // 'pre-fills from the registry prediction...' above, which asserts the
    // exact payload shape after a picker-driven Add.
  });
});
