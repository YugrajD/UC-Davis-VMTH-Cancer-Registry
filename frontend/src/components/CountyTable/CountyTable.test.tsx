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
    render(<CountyTable data={countyData} countRange={{ min: 1, max: 10 }} rateType="pccp" />);

    expect(renderedCountyOrder()).toEqual(['Charlie', 'Alpha', 'Bravo']);

    await user.click(screen.getAllByRole('columnheader')[3]);
    expect(renderedCountyOrder()).toEqual(['Bravo', 'Alpha', 'Charlie']);

    await user.click(screen.getAllByRole('columnheader')[0]);
    expect(renderedCountyOrder()).toEqual(['Charlie', 'Bravo', 'Alpha']);
  });

  it('notifies hover changes for county rows', () => {
    const onCountyHover = vi.fn();
    render(<CountyTable data={countyData} countRange={{ min: 1, max: 10 }} onCountyHover={onCountyHover} rateType="pccp" />);

    const row = screen.getByText('Alpha').closest('tr');
    if (!row) throw new Error('County row not found');

    fireEvent.mouseEnter(row);
    fireEvent.mouseLeave(row);

    expect(onCountyHover).toHaveBeenCalledWith('Alpha');
    expect(onCountyHover).toHaveBeenCalledWith(null);
  });

  it('applies selected styling to the selected county row', () => {
    render(<CountyTable data={countyData} countRange={{ min: 1, max: 10 }} selectedCounty="Bravo" rateType="pccp" />);

    const row = screen.getByText('Bravo').closest('tr');

    expect(row?.className).toContain('ring-2');
    expect(row?.className).toContain('ring-[var(--color-primary-orange)]');
  });

  it('renders Cancer Tested Positive and Total Tested column headers before PCCP', () => {
    render(<CountyTable data={countyData} countRange={{ min: 1, max: 10 }} rateType="pccp" />);

    const headers = screen.getAllByRole('columnheader').map(h => h.textContent?.trim());
    expect(headers[1]).toContain('Cancer Tested Positive');
    expect(headers[2]).toContain('Total Tested');
    expect(headers[3]).toContain('PCCP');
  });

  it('renders numerator/denominator values for counties with PCCP data', () => {
    render(<CountyTable data={pccpCountyData} countRange={{ min: 40, max: 40 }} rateType="pccp" />);

    const row = screen.getByText('Yolo').closest('tr');
    const cells = row?.querySelectorAll('td');
    expect(cells?.[1].textContent?.trim()).toBe('40');
    expect(cells?.[2].textContent?.trim()).toBe('100');
  });

  it('renders a dash for numerator/denominator when a county has no PCCP data', () => {
    render(<CountyTable data={countyData} countRange={{ min: 1, max: 10 }} rateType="pccp" />);

    const row = screen.getByText('Alpha').closest('tr');
    const cells = row?.querySelectorAll('td');
    expect(cells?.[1].textContent?.trim()).toBe('—');
    expect(cells?.[2].textContent?.trim()).toBe('—');
  });

  it('highlights the Cancer Tested Positive column header when rateType is numerator', () => {
    render(<CountyTable data={countyData} countRange={{ min: 1, max: 10 }} rateType="numerator" />);

    const headers = screen.getAllByRole('columnheader');
    expect(headers[1].className).toContain('border-[var(--color-primary-orange)]');
    expect(headers[2].className).not.toContain('border-[var(--color-primary-orange)]');
    expect(headers[3].className).not.toContain('border-[var(--color-primary-orange)]');
  });

  it('highlights the Total Tested column cells when rateType is denominator', () => {
    render(<CountyTable data={pccpCountyData} countRange={{ min: 40, max: 40 }} rateType="denominator" />);

    const row = screen.getByText('Yolo').closest('tr');
    const cells = row?.querySelectorAll('td');
    expect(cells?.[2].className).toContain('bg-[var(--color-primary-orange)]/10');
    expect(cells?.[1].className).not.toContain('bg-[var(--color-primary-orange)]/10');
  });

  it('highlights the PCCP cell with an inset border when rateType is pccp', () => {
    render(<CountyTable data={pccpCountyData} countRange={{ min: 40, max: 40 }} rateType="pccp" />);

    const row = screen.getByText('Yolo').closest('tr');
    const pccpCell = row?.querySelectorAll('td')[3] as HTMLElement;
    expect(pccpCell.style.boxShadow).toContain('var(--color-primary-orange)');
  });

  it('does not add a PCCP box-shadow when a different rateType is active', () => {
    render(<CountyTable data={pccpCountyData} countRange={{ min: 40, max: 40 }} rateType="numerator" />);

    const row = screen.getByText('Yolo').closest('tr');
    const pccpCell = row?.querySelectorAll('td')[3] as HTMLElement;
    expect(pccpCell.style.boxShadow).toBe('');
  });
});
