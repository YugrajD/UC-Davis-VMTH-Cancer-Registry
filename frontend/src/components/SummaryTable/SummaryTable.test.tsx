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
    render(<SummaryTable data={summary} rateType="pccp" />);

    expect(screen.getByText('California')).toBeInTheDocument();
    expect(screen.getByText('UC Davis Catchment Area')).toBeInTheDocument();
    expect(screen.getByText('Northern Region')).toBeInTheDocument();
    expect(screen.queryByText('Alpha')).not.toBeInTheDocument();
  });

  it('expands and collapses rows when toggle buttons are clicked', async () => {
    const user = userEvent.setup();
    render(<SummaryTable data={summary} rateType="pccp" />);

    await user.click(screen.getAllByRole('button')[0]);
    expect(screen.queryByText('UC Davis Catchment Area')).not.toBeInTheDocument();

    await user.click(screen.getAllByRole('button')[0]);
    expect(screen.getByText('UC Davis Catchment Area')).toBeInTheDocument();

    await user.click(screen.getAllByRole('button')[1]);
    expect(screen.queryByText('Northern Region')).not.toBeInTheDocument();
  });

  it('toggles global count sort ordering', async () => {
    const user = userEvent.setup();
    render(<SummaryTable data={summary} rateType="pccp" />);

    expect(visibleNames()).toEqual([
      'California',
      'UC Davis Catchment Area',
      'Northern Region',
      'Bay Region',
      'Southern Region',
    ]);

    await user.click(screen.getAllByRole('columnheader')[3]);

    expect(visibleNames()).toEqual([
      'California',
      'Southern Region',
      'UC Davis Catchment Area',
      'Bay Region',
      'Northern Region',
    ]);
  });

  it('allows per-row count sorting to override global sort for that subtree', () => {
    render(<SummaryTable data={summary} rateType="pccp" />);

    const catchmentRow = screen.getByText('UC Davis Catchment Area').closest('tr');
    const countCell = catchmentRow?.querySelectorAll('td')[3];
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

  it('renders Cancer Tested Positive and Total Tested column headers before PCCP', () => {
    render(<SummaryTable data={summary} rateType="pccp" />);

    const headers = screen.getAllByRole('columnheader').map(h => h.textContent?.trim());
    expect(headers[1]).toContain('Cancer Tested Positive');
    expect(headers[2]).toContain('Total Tested');
    expect(headers[3]).toContain('PCCP');
  });

  it('renders numerator/denominator values when present, dash otherwise', () => {
    const withPccp: RegionSummary = {
      name: 'California',
      type: 'state',
      count: 40,
      casePatients: 40,
      totalPatients: 100,
      children: [],
    };
    render(<SummaryTable data={withPccp} rateType="pccp" />);

    const row = screen.getByText('California').closest('tr');
    const cells = row?.querySelectorAll('td');
    expect(cells?.[1].textContent?.trim()).toBe('40');
    expect(cells?.[2].textContent?.trim()).toBe('100');
  });

  it('renders a dash for numerator/denominator when absent', () => {
    render(<SummaryTable data={summary} rateType="pccp" />);

    const row = screen.getByText('California').closest('tr');
    const cells = row?.querySelectorAll('td');
    expect(cells?.[1].textContent?.trim()).toBe('—');
    expect(cells?.[2].textContent?.trim()).toBe('—');
  });

  it('highlights the Cancer Tested Positive column when rateType is numerator', () => {
    render(<SummaryTable data={summary} rateType="numerator" />);

    const headers = screen.getAllByRole('columnheader');
    expect(headers[1].className).toContain('border-[var(--color-primary-orange)]');
    expect(headers[2].className).not.toContain('border-[var(--color-primary-orange)]');
    expect(headers[3].className).not.toContain('border-[var(--color-primary-orange)]');

    const row = screen.getByText('California').closest('tr');
    const cells = row?.querySelectorAll('td');
    expect(cells?.[1].className).toContain('bg-[var(--color-primary-orange)]/10');
  });

  it('highlights the PCCP column by default', () => {
    render(<SummaryTable data={summary} rateType="pccp" />);

    const headers = screen.getAllByRole('columnheader');
    expect(headers[3].className).toContain('border-[var(--color-primary-orange)]');

    const row = screen.getByText('California').closest('tr');
    const cells = row?.querySelectorAll('td');
    expect(cells?.[3].className).toContain('bg-[var(--color-primary-orange)]/10');
  });
});
