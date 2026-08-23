import { useState, type ReactNode } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AgeDetail } from '../../api/client';
import type { AgeGroup } from '../../types';
import { AgeDisparitiesView } from './AgeDisparitiesView';

const { fetchAgeDetailMock } = vi.hoisted(() => ({
  fetchAgeDetailMock: vi.fn(),
}));

vi.mock('../../api/client', () => ({
  fetchAgeDetail: fetchAgeDetailMock,
}));

vi.mock('react-simple-maps', () => ({
  ComposableMap: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  Geographies: ({
    children,
  }: {
    children: (args: { geographies: never[] }) => ReactNode;
  }) => <div>{children({ geographies: [] })}</div>,
  Geography: () => null,
}));

function ageDetail(ageGroup: string): AgeDetail {
  return {
    age_group: ageGroup,
    total_cases: 1,
    age_total_patients: 2,
    global_total_patients: 10,
    pccp_within_age: 50,
    pccp_of_all: 10,
    sex_breakdown: [],
    cancer_types: [],
    county_cases: [],
  };
}

function Harness() {
  const [visible, setVisible] = useState(true);
  const [selectedAgeGroup, setSelectedAgeGroup] = useState<AgeGroup | ''>('');

  return (
    <>
      <button onClick={() => setVisible(current => !current)}>Toggle age tab</button>
      {visible && (
        <AgeDisparitiesView
          selectedAgeGroup={selectedAgeGroup}
          onSelectedAgeGroupChange={setSelectedAgeGroup}
        />
      )}
    </>
  );
}

describe('AgeDisparitiesView', () => {
  beforeEach(() => {
    fetchAgeDetailMock.mockReset();
    fetchAgeDetailMock.mockImplementation(async (ageGroup: string) => ageDetail(ageGroup));
  });

  it('defaults to the first age group and preserves the selection across remounts', async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const select = await screen.findByLabelText('Select Age Group:');
    await waitFor(() => expect(select).toHaveValue('young'));
    await waitFor(() => expect(fetchAgeDetailMock).toHaveBeenCalledWith('young'));

    await user.selectOptions(select, 'adult');
    await waitFor(() => expect(fetchAgeDetailMock).toHaveBeenCalledWith('adult'));

    await user.click(screen.getByRole('button', { name: 'Toggle age tab' }));
    expect(screen.queryByLabelText('Select Age Group:')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Toggle age tab' }));
    expect(await screen.findByLabelText('Select Age Group:')).toHaveValue('adult');
  });

  it('uses the absolute PCCP legend and explains overlapping diagnoses', async () => {
    render(<Harness />);

    expect(await screen.findByText('0%')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
    expect(screen.getByText(/a patient may have multiple case diagnoses/i)).toBeInTheDocument();
  });
});
