import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { FilterState } from '../../types';
import { Filters } from './Filters';

vi.mock('../../api/client', () => ({
  fetchFilterOptions: vi.fn().mockResolvedValue({
    cancer_types: [
      { name: 'Lymphoma' },
      { name: 'Mast Cell Tumor' },
    ],
    breeds: [
      { name: 'Golden Retriever' },
      { name: 'Labrador Retriever' },
    ],
    species: [],
  }),
}));

const defaultFilters: FilterState = {
  rateType: 'incidence',
  sex: 'all',
  ageGroup: 'all',
  cancerType: 'All Types',
  breed: 'All Breeds',
};

describe('Filters', () => {
  it('emits a full updated FilterState without dropping existing fields', async () => {
    const user = userEvent.setup();
    const onFilterChange = vi.fn();
    render(<Filters filters={defaultFilters} onFilterChange={onFilterChange} />);

    // Dropdowns: [rateType, sex, ageGroup, cancerType, breed, yearStart, yearEnd]
    const selects = screen.getAllByRole('combobox');
    const sexSelect = selects[1];
    const cancerSelect = selects[3];
    const breedSelect = selects[4];

    await user.selectOptions(sexSelect, 'male_neutered');
    await user.selectOptions(cancerSelect, await screen.findByRole('option', { name: 'Lymphoma' }));
    await user.selectOptions(breedSelect, await screen.findByRole('option', { name: 'Golden Retriever' }));

    expect(onFilterChange).toHaveBeenNthCalledWith(1, {
      ...defaultFilters,
      sex: 'male_neutered',
    });
    expect(onFilterChange).toHaveBeenNthCalledWith(2, {
      ...defaultFilters,
      cancerType: 'Lymphoma',
    });
    expect(onFilterChange).toHaveBeenNthCalledWith(3, {
      ...defaultFilters,
      breed: 'Golden Retriever',
    });
  });

  it('renders active filter summary as None by default', () => {
    render(<Filters filters={defaultFilters} onFilterChange={vi.fn()} />);

    expect(screen.getByText('None')).toBeInTheDocument();
  });

  it('renders active cancer type, breed, and sex summaries', () => {
    render(
      <Filters
        filters={{
          ...defaultFilters,
          cancerType: 'Lymphoma',
          breed: 'Golden Retriever',
          sex: 'female_spayed',
        }}
        onFilterChange={vi.fn()}
      />,
    );

    expect(screen.getByText(/active filters:/i).closest('p')).toHaveTextContent(
      'Active filters: Lymphoma, Golden Retriever, Female Spayed',
    );
  });
});
