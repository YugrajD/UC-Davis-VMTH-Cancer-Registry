import type { FilterState, Sex, RateType } from '../../types';
import { CANCER_TYPES, BREEDS, SEX_OPTIONS, RATE_OPTIONS } from '../../types';

interface FiltersProps {
  filters: FilterState;
  onFilterChange: (filters: FilterState) => void;
}

export function Filters({ filters, onFilterChange }: FiltersProps) {
  const handleChange = (key: keyof FilterState, value: string) => {
    onFilterChange({
      ...filters,
      [key]: value,
    });
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4 uppercase tracking-wider">
        Filters
      </h3>
      
      <div className="space-y-4">
        {/* Rate Type */}
        <div>
          <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5 uppercase tracking-wide">
            Rate
          </label>
          <select
            value={filters.rateType}
            onChange={(e) => handleChange('rateType', e.target.value as RateType)}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white 
                       focus:ring-2 focus:ring-[var(--color-teal)] focus:border-[var(--color-teal)]
                       transition-colors duration-150"
          >
            {RATE_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Sex */}
        <div>
          <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5 uppercase tracking-wide">
            Sex
          </label>
          <select
            value={filters.sex}
            onChange={(e) => handleChange('sex', e.target.value as Sex)}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white 
                       focus:ring-2 focus:ring-[var(--color-teal)] focus:border-[var(--color-teal)]
                       transition-colors duration-150"
          >
            {SEX_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Cancer Type */}
        <div>
          <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5 uppercase tracking-wide">
            Cancer Type
          </label>
          <select
            value={filters.cancerType}
            onChange={(e) => handleChange('cancerType', e.target.value)}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white 
                       focus:ring-2 focus:ring-[var(--color-teal)] focus:border-[var(--color-teal)]
                       transition-colors duration-150"
          >
            {CANCER_TYPES.map(type => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>

        {/* Breed */}
        <div>
          <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5 uppercase tracking-wide">
            Breed
          </label>
          <select
            value={filters.breed}
            onChange={(e) => handleChange('breed', e.target.value)}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white 
                       focus:ring-2 focus:ring-[var(--color-teal)] focus:border-[var(--color-teal)]
                       transition-colors duration-150"
          >
            {BREEDS.map(breed => (
              <option key={breed} value={breed}>
                {breed}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Active Filters Summary */}
      <div className="mt-4 pt-4 border-t border-gray-100">
        <p className="text-xs text-[var(--color-text-secondary)]">
          <span className="font-medium">Active filters:</span>{' '}
          {filters.cancerType !== 'All Types' && <span className="text-[var(--color-teal)]">{filters.cancerType}</span>}
          {filters.breed !== 'All Breeds' && <span className="text-[var(--color-teal)]">{filters.cancerType !== 'All Types' ? ', ' : ''}{filters.breed}</span>}
          {filters.sex !== 'all' && <span className="text-[var(--color-teal)]">{(filters.cancerType !== 'All Types' || filters.breed !== 'All Breeds') ? ', ' : ''}{SEX_OPTIONS.find(s => s.value === filters.sex)?.label}</span>}
          {filters.cancerType === 'All Types' && filters.breed === 'All Breeds' && filters.sex === 'all' && <span className="italic">None</span>}
        </p>
      </div>
    </div>
  );
}
