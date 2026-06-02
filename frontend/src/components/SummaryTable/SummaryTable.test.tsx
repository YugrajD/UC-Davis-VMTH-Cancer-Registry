import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import type { RegionSummary } from '../../types';
import { SummaryTable } from './SummaryTable';

const summary: RegionSummary = {
  name: 'California',
  type: 'state',
  count: 100,
  children: [
    {
      name: 'UC Davis Catchment Area',
      type: 'catchment',
      count: 80,
      children: [
        {
          name: 'Northern Region',
          type: 'region',
          count: 50,
          children: [
            { name: 'Alpha', type: 'county', count: 30 },
            { name: 'Bravo', type: 'county', count: 20 },
          ],
        },
        {
          name: 'Bay Region',
          type: 'region',
          count: 30,
          children: [
            { name: 'Delta', type: 'county', count: 30 },
          ],
        },
      ],
    },
    {
      name: 'Southern Region',
      type: 'region',
      count: 20,
      children: [
        { name: 'Charlie', type: 'county', count: 20 },
      ],
    },
  ],
};

function visibleNames() {
  return screen.getAllByRole('row').slice(1).map((row) => {
    const firstCell = row.querySelector('td');
    return firstCell?.textContent?.trim() ?? '';
  });
}

describe('SummaryTable', () => {
  it('initially expands California and UC Davis Catchment Area', () => {
    render(<SummaryTable data={summary} />);

    expect(screen.getByText('California')).toBeInTheDocument();
    expect(screen.getByText('UC Davis Catchment Area')).toBeInTheDocument();
    expect(screen.getByText('Northern Region')).toBeInTheDocument();
    expect(screen.queryByText('Alpha')).not.toBeInTheDocument();
  });

  it('expands and collapses rows when toggle buttons are clicked', async () => {
    const user = userEvent.setup();
    render(<SummaryTable data={summary} />);

    await user.click(screen.getAllByRole('button')[0]);
    expect(screen.queryByText('UC Davis Catchment Area')).not.toBeInTheDocument();

    await user.click(screen.getAllByRole('button')[0]);
    expect(screen.getByText('UC Davis Catchment Area')).toBeInTheDocument();

    await user.click(screen.getAllByRole('button')[1]);
    expect(screen.queryByText('Northern Region')).not.toBeInTheDocument();
  });

  it('toggles global count sort ordering', async () => {
    const user = userEvent.setup();
    render(<SummaryTable data={summary} />);

    expect(visibleNames()).toEqual([
      'California',
      'UC Davis Catchment Area',
      'Northern Region',
      'Bay Region',
      'Southern Region',
    ]);

    await user.click(screen.getAllByRole('columnheader')[1]);

    expect(visibleNames()).toEqual([
      'California',
      'Southern Region',
      'UC Davis Catchment Area',
      'Bay Region',
      'Northern Region',
    ]);
  });

  it('allows per-row count sorting to override global sort for that subtree', () => {
    render(<SummaryTable data={summary} />);

    const catchmentRow = screen.getByText('UC Davis Catchment Area').closest('tr');
    const countCell = catchmentRow?.querySelectorAll('td')[1];
    if (!countCell) throw new Error('Catchment count cell missing');

    fireEvent.click(countCell);

    expect(visibleNames()).toEqual([
      'California',
      'UC Davis Catchment Area',
      'Bay Region',
      'Northern Region',
      'Southern Region',
    ]);
  });
});
