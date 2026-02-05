import { useState, useMemo } from 'react';
import type { CountyData } from '../../types';
import { scaleLinear } from 'd3-scale';

interface CountyTableProps {
  data: CountyData[];
  rateRange: { min: number; max: number };
  onCountyHover?: (county: string | null) => void;
  selectedCounty?: string | null;
}

type SortField = 'county' | 'count' | 'population' | 'rate';
type SortDirection = 'asc' | 'desc';

export function CountyTable({ data, rateRange, onCountyHover, selectedCounty }: CountyTableProps) {
  const [sortField, setSortField] = useState<SortField>('rate');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const colorScale = useMemo(() => {
    return scaleLinear<string>()
      .domain([rateRange.min, rateRange.max])
      .range(['#E6F3F5', '#1A6B77']);
  }, [rateRange]);

  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'county':
          comparison = a.county.localeCompare(b.county);
          break;
        case 'count':
          comparison = a.count - b.count;
          break;
        case 'population':
          comparison = a.population - b.population;
          break;
        case 'rate':
          comparison = a.rate - b.rate;
          break;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [data, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => (
    <svg
      className={`w-3 h-3 ml-1 inline-block transition-transform ${
        sortField === field 
          ? sortDirection === 'asc' ? 'rotate-180' : '' 
          : 'opacity-30'
      }`}
      fill="currentColor"
      viewBox="0 0 20 20"
    >
      <path
        fillRule="evenodd"
        d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
        clipRule="evenodd"
      />
    </svg>
  );

  const getCellColor = (rate: number) => {
    const bg = colorScale(rate);
    // Calculate luminance to determine text color
    const hex = bg.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16);
    const g = parseInt(hex.substr(2, 2), 16);
    const b = parseInt(hex.substr(4, 2), 16);
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    const textColor = luminance > 0.5 ? '#333333' : '#FFFFFF';
    
    return { backgroundColor: bg, color: textColor };
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
          County Details
        </h3>
        <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
          Click column headers to sort • Hover rows to highlight on map
        </p>
      </div>
      
      <div className="max-h-[400px] overflow-y-auto custom-scrollbar">
        <table className="w-full text-left">
          <thead className="sticky top-0 z-10">
            <tr className="bg-[var(--color-teal)] text-white">
              <th 
                className="py-2.5 px-3 text-xs font-semibold uppercase tracking-wider cursor-pointer hover:bg-[var(--color-teal-dark)] transition-colors"
                onClick={() => handleSort('county')}
              >
                County <SortIcon field="county" />
              </th>
              <th 
                className="py-2.5 px-3 text-xs font-semibold uppercase tracking-wider text-right cursor-pointer hover:bg-[var(--color-teal-dark)] transition-colors"
                onClick={() => handleSort('count')}
              >
                Count <SortIcon field="count" />
              </th>
              <th 
                className="py-2.5 px-3 text-xs font-semibold uppercase tracking-wider text-right cursor-pointer hover:bg-[var(--color-teal-dark)] transition-colors"
                onClick={() => handleSort('population')}
              >
                Pop <SortIcon field="population" />
              </th>
              <th 
                className="py-2.5 px-3 text-xs font-semibold uppercase tracking-wider text-right cursor-pointer hover:bg-[var(--color-teal-dark)] transition-colors"
                onClick={() => handleSort('rate')}
              >
                Rate <SortIcon field="rate" />
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedData.map((county) => {
              const rateStyle = getCellColor(county.rate);
              const isSelected = selectedCounty === county.county;
              
              return (
                <tr 
                  key={county.county}
                  className={`border-b border-gray-100 transition-all duration-150 cursor-pointer
                    ${isSelected ? 'ring-2 ring-[var(--color-primary-orange)] ring-inset' : 'hover:ring-1 hover:ring-[var(--color-teal-light)] hover:ring-inset'}
                  `}
                  onMouseEnter={() => onCountyHover?.(county.county)}
                  onMouseLeave={() => onCountyHover?.(null)}
                >
                  <td className="py-2 px-3 text-sm font-medium text-[var(--color-text-primary)]">
                    {county.county}
                    <span className="text-xs text-[var(--color-text-secondary)] ml-1.5">
                      ({county.region})
                    </span>
                  </td>
                  <td className="py-2 px-3 text-sm text-right tabular-nums" style={getCellColor(county.count / data.reduce((max, c) => Math.max(max, c.count), 0) * rateRange.max)}>
                    {county.count.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-sm text-right tabular-nums text-[var(--color-text-secondary)]">
                    {county.population.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-sm text-right tabular-nums font-semibold" style={rateStyle}>
                    {county.rate.toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      
      {/* Summary footer */}
      <div className="px-4 py-2 bg-gray-50 border-t border-gray-200 flex justify-between items-center">
        <span className="text-xs text-[var(--color-text-secondary)]">
          Showing {data.length} counties
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-[var(--color-text-secondary)]">Rate:</span>
          <div className="flex items-center">
            <div className="w-20 h-3 rounded" style={{ background: 'linear-gradient(to right, #E6F3F5, #1A6B77)' }} />
          </div>
          <span className="text-xs text-[var(--color-text-secondary)]">
            {rateRange.min.toFixed(1)} - {rateRange.max.toFixed(1)}
          </span>
        </div>
      </div>
    </div>
  );
}
