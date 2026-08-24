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

const pccpCountyData: CountyData[] = [
  { county: 'Yolo', region: 'Sacramento Valley', count: 40, fips: '057', casePatients: 40, totalPatients: 100 },
  { county: 'Placer', region: 'Sierra Nevada', count: 10, fips: '061', casePatients: 5, totalPatients: 200 },
];

function renderedCountyOrder() {
  return screen.getAllByRole('row').slice(1).map((row) => {
    const firstCell = row.querySelector('td');
    return firstCell?.textContent?.trim().replace(/\s*\(.+\)$/, '') ?? '';
  });
}

describe('CountyTable', () => {
  it('defaults to value descending, toggles ascending, and switches county sort to descending first', async () => {
    const user = userEvent.setup();
    render(<CountyTable data={countyData} rateType="pccp" />);

    expect(renderedCountyOrder()).toEqual(['Charlie', 'Alpha', 'Bravo']);

    await user.click(screen.getAllByRole('columnheader')[1]);
    expect(renderedCountyOrder()).toEqual(['Bravo', 'Alpha', 'Charlie']);

    await user.click(screen.getAllByRole('columnheader')[0]);
    expect(renderedCountyOrder()).toEqual(['Charlie', 'Bravo', 'Alpha']);
  });

  it('notifies hover changes for county rows', () => {
    const onCountyHover = vi.fn();
    render(<CountyTable data={countyData} onCountyHover={onCountyHover} rateType="pccp" />);

    const row = screen.getByText('Alpha').closest('tr');
    if (!row) throw new Error('County row not found');

    fireEvent.mouseEnter(row);
    fireEvent.mouseLeave(row);

    expect(onCountyHover).toHaveBeenCalledWith('Alpha');
    expect(onCountyHover).toHaveBeenCalledWith(null);
  });

  it('applies selected styling to the selected county row', () => {
    render(<CountyTable data={countyData} selectedCounty="Bravo" rateType="pccp" />);

    const row = screen.getByText('Bravo').closest('tr');

    expect(row?.className).toContain('ring-2');
    expect(row?.className).toContain('ring-[var(--color-primary-orange)]');
  });

  it('shows only a single value column matching the Rate filter, not all three', () => {
    render(<CountyTable data={countyData} rateType="pccp" />);

    const headers = screen.getAllByRole('columnheader').map(h => h.textContent?.trim());
    expect(headers).toHaveLength(2);
    expect(headers[0]).toContain('County');
    expect(headers[1]).toContain('PCCP');
  });

  it('renders PCCP values and colors by PCCP when rateType is pccp', () => {
    render(<CountyTable data={pccpCountyData} rateType="pccp" />);

    const headers = screen.getAllByRole('columnheader').map(h => h.textContent?.trim());
    expect(headers[1]).toContain('PCCP');

    const row = screen.getByText('Yolo').closest('tr');
    const cell = row?.querySelectorAll('td')[1];
    expect(cell?.textContent?.trim()).toBe('40.0');
  });

  it('switches to Cancer Tested Positive column and values when rateType is numerator', () => {
    render(<CountyTable data={pccpCountyData} rateType="numerator" />);

    const headers = screen.getAllByRole('columnheader').map(h => h.textContent?.trim());
    expect(headers[1]).toContain('Cancer Tested Positive');
    expect(headers[1]).not.toContain('Total Tested');
    expect(headers[1]).not.toContain('PCCP');

    const row = screen.getByText('Yolo').closest('tr');
    const cell = row?.querySelectorAll('td')[1];
    expect(cell?.textContent?.trim()).toBe('40');
  });

  it('switches to Total Tested column and values when rateType is denominator', () => {
    render(<CountyTable data={pccpCountyData} rateType="denominator" />);

    const headers = screen.getAllByRole('columnheader').map(h => h.textContent?.trim());
    expect(headers[1]).toContain('Total Tested');

    const row = screen.getByText('Yolo').closest('tr');
    const cell = row?.querySelectorAll('td')[1];
    expect(cell?.textContent?.trim()).toBe('100');
  });

  it('sorts by the currently selected metric, not always PCCP', () => {
    // Placer has a lower PCCP (10) than Yolo (40), but a higher denominator
    // (200 vs 100) — denominator-mode sort should rank Placer first.
    render(<CountyTable data={pccpCountyData} rateType="denominator" />);

    expect(renderedCountyOrder()).toEqual(['Placer', 'Yolo']);
  });

  it('renders a dash when a county has no data for the selected metric', () => {
    render(<CountyTable data={countyData} rateType="numerator" />);

    const row = screen.getByText('Alpha').closest('tr');
    const cell = row?.querySelectorAll('td')[1];
    expect(cell?.textContent?.trim()).toBe('—');
  });
});
