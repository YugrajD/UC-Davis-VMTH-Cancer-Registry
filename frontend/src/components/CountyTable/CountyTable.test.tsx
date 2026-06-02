import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { CountyData } from '../../types';
import { CountyTable } from './CountyTable';

const countyData: CountyData[] = [
  { county: 'Alpha', region: 'North', count: 5, fips: '001' },
  { county: 'Charlie', region: 'South', count: 10, fips: '003' },
  { county: 'Bravo', region: 'Central', count: 1, fips: '002' },
];

function renderedCountyOrder() {
  return screen.getAllByRole('row').slice(1).map((row) => {
    const firstCell = row.querySelector('td');
    return firstCell?.textContent?.trim().replace(/\s*\(.+\)$/, '') ?? '';
  });
}

describe('CountyTable', () => {
  it('defaults to count descending, toggles count ascending, and switches county sort to descending first', async () => {
    const user = userEvent.setup();
    render(<CountyTable data={countyData} countRange={{ min: 1, max: 10 }} />);

    expect(renderedCountyOrder()).toEqual(['Charlie', 'Alpha', 'Bravo']);

    await user.click(screen.getAllByRole('columnheader')[1]);
    expect(renderedCountyOrder()).toEqual(['Bravo', 'Alpha', 'Charlie']);

    await user.click(screen.getAllByRole('columnheader')[0]);
    expect(renderedCountyOrder()).toEqual(['Charlie', 'Bravo', 'Alpha']);
  });

  it('notifies hover changes for county rows', () => {
    const onCountyHover = vi.fn();
    render(<CountyTable data={countyData} countRange={{ min: 1, max: 10 }} onCountyHover={onCountyHover} />);

    const row = screen.getByText('Alpha').closest('tr');
    if (!row) throw new Error('County row not found');

    fireEvent.mouseEnter(row);
    fireEvent.mouseLeave(row);

    expect(onCountyHover).toHaveBeenCalledWith('Alpha');
    expect(onCountyHover).toHaveBeenCalledWith(null);
  });

  it('applies selected styling to the selected county row', () => {
    render(<CountyTable data={countyData} countRange={{ min: 1, max: 10 }} selectedCounty="Bravo" />);

    const row = screen.getByText('Bravo').closest('tr');

    expect(row?.className).toContain('ring-2');
    expect(row?.className).toContain('ring-[var(--color-primary-orange)]');
  });
});
