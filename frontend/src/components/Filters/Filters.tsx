import { useEffect, useState } from 'react';
import type { FilterState, Sex, AgeGroup, RateType } from '../../types';
import { SEX_OPTIONS, AGE_GROUP_OPTIONS, RATE_OPTIONS } from '../../types';
import { fetchFilterOptions } from '../../api/client';

interface FiltersProps {
  filters: FilterState;
  onFilterChange: (filters: FilterState) => void;
}

export function Filters({ filters, onFilterChange }: FiltersProps) {
  const [cancerTypes, setCancerTypes] = useState<string[]>(['All Types']);
  const [breeds, setBreeds] = useState<string[]>(['All Breeds']);
  const [yearOptions, setYearOptions] = useState<number[]>([]);

  useEffect(() => {
    fetchFilterOptions()
      .then(opts => {
        const names = opts.cancer_types.map(ct => ct.name).sort();
        setCancerTypes(['All Types', ...names]);
        const breedNames = opts.breeds.map(b => b.name).sort();
        setBreeds(['All Breeds', ...breedNames]);
        const [min, max] = opts.year_range;
        const years: number[] = [];
        for (let y = min; y <= max; y++) years.push(y);
        setYearOptions(years);
      })
      .catch(() => {});
  }, []);

  const handleChange = (key: keyof FilterState, value: string) => {
    onFilterChange({
      ...filters,
      [key]: value,
    });
  };

  const handleYearChange = (key: 'yearStart' | 'yearEnd', value: string) => {
    onFilterChange({
      ...filters,
      [key]: value ? Number(value) : undefined,
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

        {/* Age */}
        <div>
          <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5 uppercase tracking-wide">
            Age at Diagnosis
          </label>
          <select
            value={filters.ageGroup}
            onChange={(e) => handleChange('ageGroup', e.target.value as AgeGroup)}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white
                       focus:ring-2 focus:ring-[var(--color-teal)] focus:border-[var(--color-teal)]
                       transition-colors duration-150"
          >
            {AGE_GROUP_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}{option.range ? ` (${option.range})` : ''}
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
            {cancerTypes.map(type => (
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
            {breeds.map(breed => (
              <option key={breed} value={breed}>
                {breed}
              </option>
            ))}
          </select>
        </div>

        {/* Year Start */}
        {yearOptions.length > 0 && (
          <div>
            <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5 uppercase tracking-wide">
              Year Start
            </label>
            <select
              value={filters.yearStart ?? ''}
              onChange={(e) => handleYearChange('yearStart', e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white
                         focus:ring-2 focus:ring-[var(--color-teal)] focus:border-[var(--color-teal)]
                         transition-colors duration-150"
            >
              <option value="">All Years</option>
              {yearOptions
                .filter(y => !filters.yearEnd || y <= filters.yearEnd)
                .map(y => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
        )}

        {/* Year End */}
        {yearOptions.length > 0 && (
          <div>
            <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5 uppercase tracking-wide">
              Year End
            </label>
            <select
              value={filters.yearEnd ?? ''}
              onChange={(e) => handleYearChange('yearEnd', e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white
                         focus:ring-2 focus:ring-[var(--color-teal)] focus:border-[var(--color-teal)]
                         transition-colors duration-150"
            >
              <option value="">All Years</option>
              {yearOptions
                .filter(y => !filters.yearStart || y >= filters.yearStart)
                .map(y => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
        )}
      </div>

      {/* Active Filters Summary */}
      <div className="mt-4 pt-4 border-t border-gray-100">
        <p className="text-xs text-[var(--color-text-secondary)]">
          <span className="font-medium">Active filters:</span>{' '}
          {(() => {
            const parts: string[] = [];
            if (filters.cancerType !== 'All Types') parts.push(filters.cancerType);
            if (filters.breed !== 'All Breeds') parts.push(filters.breed);
            if (filters.sex !== 'all') parts.push(SEX_OPTIONS.find(s => s.value === filters.sex)?.label ?? '');
            if (filters.ageGroup !== 'all') {
              const ag = AGE_GROUP_OPTIONS.find(a => a.value === filters.ageGroup);
              if (ag) parts.push(`${ag.label} (${ag.range})`);
            }
            if (filters.yearStart && filters.yearEnd) parts.push(`${filters.yearStart}–${filters.yearEnd}`);
            else if (filters.yearStart) parts.push(`from ${filters.yearStart}`);
            else if (filters.yearEnd) parts.push(`to ${filters.yearEnd}`);
            return parts.length > 0
              ? <span className="text-[var(--color-teal)]">{parts.join(', ')}</span>
              : <span className="italic">None</span>;
          })()}
        </p>
      </div>
    </div>
  );
}
