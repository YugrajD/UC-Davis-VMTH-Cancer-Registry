import { useEffect, useMemo, useRef, useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import { fetchFilterOptions, fetchBreedDetail } from '../../api/client';
import type { BreedDetail, FilterOptions } from '../../api/client';

const GEO_URL =
  'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson';

const MAP_PROJECTION_CONFIG = {
  scale: 2400,
  center: [-119.5, 37.5] as [number, number],
};

export function BreedDisparitiesView() {
  const [breeds, setBreeds] = useState<string[]>([]);
  const [selectedBreed, setSelectedBreed] = useState<string>('');
  const [detail, setDetail] = useState<BreedDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Autocomplete state
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // Tooltip for map
  const [tooltip, setTooltip] = useState<{
    county: string;
    count: number;
    x: number;
    y: number;
  } | null>(null);

  // Load breed list from filters endpoint
  useEffect(() => {
    fetchFilterOptions()
      .then((opts: FilterOptions) => {
        const breedNames = opts.breeds.map((b) => b.name).sort();
        setBreeds(breedNames);
        if (breedNames.length > 0) {
          setSelectedBreed(breedNames[0]);
          setQuery(breedNames[0]);
        }
      })
      .catch(() => setError('Failed to load breed list'));
  }, []);

  // Fetch breed detail whenever selected breed changes
  useEffect(() => {
    if (!selectedBreed) return;
    setLoading(true);
    setError(null);
    fetchBreedDetail(selectedBreed)
      .then((data) => {
        setDetail(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [selectedBreed]);

  // Filtered suggestions
  const suggestions = useMemo(() => {
    if (!query.trim()) return breeds.slice(0, 20);
    const lower = query.toLowerCase();
    return breeds.filter((b) => b.toLowerCase().includes(lower)).slice(0, 20);
  }, [query, breeds]);

  // Reset highlight when suggestions change
  useEffect(() => {
    setHighlightedIndex(0);
  }, [suggestions]);

  // Scroll highlighted item into view
  useEffect(() => {
    if (isOpen && listRef.current) {
      const item = listRef.current.children[highlightedIndex] as HTMLElement | undefined;
      item?.scrollIntoView({ block: 'nearest' });
    }
  }, [highlightedIndex, isOpen]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (inputRef.current && !inputRef.current.parentElement?.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const selectBreed = (breed: string) => {
    setSelectedBreed(breed);
    setQuery(breed);
    setIsOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        setIsOpen(true);
        e.preventDefault();
      }
      return;
    }
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setHighlightedIndex((i) => Math.min(i + 1, suggestions.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setHighlightedIndex((i) => Math.max(i - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        if (suggestions[highlightedIndex]) {
          selectBreed(suggestions[highlightedIndex]);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        break;
    }
  };

  // Map helpers
  const countyMap = useMemo(() => {
    const m = new Map<string, number>();
    detail?.county_cases.forEach((c) => m.set(c.county_name.toLowerCase(), c.count));
    return m;
  }, [detail]);

  const countRange = useMemo(() => {
    if (!detail || detail.county_cases.length === 0) return { min: 0, max: 1 };
    const counts = detail.county_cases.map((c) => c.count);
    return { min: Math.min(...counts), max: Math.max(...counts) };
  }, [detail]);

  const colorScale = useMemo(() => {
    return scaleLinear<string>()
      .domain([countRange.min, (countRange.min + countRange.max) / 2, countRange.max])
      .range(['#E6F3F5', '#6BB5BF', '#1A6B77']);
  }, [countRange]);

  // Bar chart max
  const maxCancerCount = detail?.cancer_types[0]?.count || 1;

  return (
    <div className="space-y-6">
      {/* Breed Autocomplete */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 flex items-center gap-4">
        <label
          htmlFor="breed-input"
          className="text-sm font-medium text-[var(--color-text-primary)] whitespace-nowrap"
        >
          Select Breed:
        </label>
        <div className="relative w-80">
          <input
            ref={inputRef}
            id="breed-input"
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setIsOpen(true);
            }}
            onFocus={() => setIsOpen(true)}
            onKeyDown={handleKeyDown}
            placeholder="Type to search breeds..."
            autoComplete="off"
            className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
          />
          {isOpen && suggestions.length > 0 && (
            <ul
              ref={listRef}
              className="absolute z-40 mt-1 w-full max-h-60 overflow-auto bg-white border border-gray-200 rounded-md shadow-lg"
            >
              {suggestions.map((breed, i) => (
                <li
                  key={breed}
                  onMouseDown={() => selectBreed(breed)}
                  onMouseEnter={() => setHighlightedIndex(i)}
                  className={`px-3 py-2 text-sm cursor-pointer ${
                    i === highlightedIndex
                      ? 'bg-[var(--color-teal)] text-white'
                      : 'text-[var(--color-text-primary)] hover:bg-gray-50'
                  } ${breed === selectedBreed ? 'font-semibold' : ''}`}
                >
                  {breed}
                </li>
              ))}
            </ul>
          )}
          {isOpen && query.trim() && suggestions.length === 0 && (
            <div className="absolute z-40 mt-1 w-full bg-white border border-gray-200 rounded-md shadow-lg px-3 py-2">
              <p className="text-sm text-[var(--color-text-secondary)]">No breeds found</p>
            </div>
          )}
        </div>
      </div>

      {/* Loading / Error */}
      {loading && (
        <div className="bg-white rounded-lg border border-gray-200 p-12 flex flex-col items-center justify-center">
          <div className="w-8 h-8 border-4 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
          <p className="mt-4 text-sm text-[var(--color-text-secondary)]">
            Loading breed data...
          </p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
          <svg
            className="w-5 h-5 text-red-500 flex-shrink-0"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {!loading && !error && detail && (
        <>
          {/* Summary Stats */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
              <span className="font-semibold text-[var(--color-text-primary)]">
                {detail.breed}
              </span>{' '}
              has{' '}
              <span className="font-semibold text-[var(--color-teal-dark)]">
                {detail.total_cases.toLocaleString()}
              </span>{' '}
              total case diagnoses across{' '}
              <span className="font-semibold">{detail.county_cases.length}</span> counties.
              {detail.sex_breakdown.length > 0 && (
                <>
                  {' '}
                  Sex distribution:{' '}
                  {detail.sex_breakdown
                    .map((s) => `${s.sex} (${s.count})`)
                    .join(', ')}
                  .
                </>
              )}
            </p>
          </div>

          {/* Two-column layout: bar chart + map */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Cancer Type Breakdown */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                  Cancer Type Breakdown
                </h3>
                <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                  Top cancer types for {detail.breed}
                </p>
              </div>
              <div className="p-6">
                {detail.cancer_types.length === 0 ? (
                  <p className="text-sm text-[var(--color-text-secondary)]">
                    No cancer type data available for this breed.
                  </p>
                ) : (
                  <div className="space-y-4">
                    {detail.cancer_types.slice(0, 15).map((ct) => {
                      const width = Math.max(5, (ct.count / maxCancerCount) * 100);
                      return (
                        <div key={ct.cancer_type} className="flex items-center gap-4">
                          <span className="w-48 text-sm text-[var(--color-text-primary)] truncate">
                            {ct.cancer_type}
                          </span>
                          <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-[var(--color-teal)] to-[var(--color-teal-dark)] rounded-full flex items-center justify-end pr-2"
                              style={{ width: `${width}%` }}
                            >
                              <span className="text-xs font-semibold text-white">
                                {ct.count.toLocaleString()}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Right: County Distribution Map */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                  County Distribution
                </h3>
                <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                  Where {detail.breed} cases are located
                </p>
              </div>
              <div className="relative" style={{ minHeight: '400px', backgroundColor: '#f8fafc' }}>
                <ComposableMap
                  projection="geoMercator"
                  projectionConfig={MAP_PROJECTION_CONFIG}
                  width={400}
                  height={400}
                  style={{ width: '100%', height: '100%' }}
                >
                  <Geographies geography={GEO_URL}>
                    {({ geographies }) =>
                      geographies.map((geo) => {
                        const name = (geo.properties.name || '') as string;
                        const count = countyMap.get(name.toLowerCase()) ?? 0;
                        const fill = count > 0 ? colorScale(count) : '#E5E7EB';

                        return (
                          <Geography
                            key={geo.rsmKey}
                            geography={geo}
                            fill={fill}
                            stroke="#FFFFFF"
                            strokeWidth={0.5}
                            style={{
                              default: { outline: 'none' },
                              hover: {
                                fill: '#F5A623',
                                stroke: '#E87722',
                                strokeWidth: 1.5,
                                outline: 'none',
                                cursor: 'pointer',
                              },
                              pressed: { fill: '#E87722', outline: 'none' },
                            }}
                            onMouseEnter={(e) => {
                              const event = e as unknown as React.MouseEvent;
                              setTooltip({
                                county: name,
                                count,
                                x: event.clientX,
                                y: event.clientY,
                              });
                            }}
                            onMouseLeave={() => setTooltip(null)}
                          />
                        );
                      })
                    }
                  </Geographies>
                </ComposableMap>

                {/* Legend */}
                <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
                  <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">
                    Cases
                  </p>
                  <div
                    className="w-28 h-3 rounded"
                    style={{
                      background:
                        'linear-gradient(to right, #E6F3F5, #6BB5BF, #1A6B77)',
                    }}
                  />
                  <div className="flex justify-between mt-1">
                    <span className="text-[10px] text-[var(--color-text-secondary)]">
                      {countRange.min}
                    </span>
                    <span className="text-[10px] text-[var(--color-text-secondary)]">
                      {countRange.max}
                    </span>
                  </div>
                  <div className="mt-2 pt-2 border-t border-gray-100 flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
                    <span className="text-[10px] text-[var(--color-text-secondary)]">
                      No data
                    </span>
                  </div>
                </div>

                {/* Tooltip */}
                {tooltip && (
                  <div
                    className="fixed z-50 pointer-events-none"
                    style={{
                      left: tooltip.x + 12,
                      top: tooltip.y - 12,
                      transform: 'translateY(-100%)',
                    }}
                  >
                    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px]">
                      <p className="font-semibold text-sm text-[var(--color-text-primary)]">
                        {tooltip.county}
                      </p>
                      <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                        {tooltip.count > 0
                          ? `${tooltip.count.toLocaleString()} cases`
                          : 'No data'}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
