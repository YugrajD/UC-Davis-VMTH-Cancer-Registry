import { useEffect, useMemo, useRef, useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import { fetchFilterOptions, fetchBreedDetail } from '../../api/client';
import type { BreedDetail } from '../../api/client';

const GEO_URL =
  'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson';

const MAP_PROJECTION_CONFIG = {
  scale: 1800,
  center: [-119.5, 37.5] as [number, number],
};

type MapMode = 'within_breed' | 'of_all';

interface BreedDisparitiesViewProps {
  selectedBreed: string;
  onSelectedBreedChange: (breed: string) => void;
}

export function BreedDisparitiesView({
  selectedBreed,
  onSelectedBreedChange,
}: BreedDisparitiesViewProps) {
  const [breeds, setBreeds] = useState<string[]>([]);
  const [loadedBreed, setLoadedBreed] = useState<string>('');
  const [detail, setDetail] = useState<BreedDetail | null>(null);
  const [loadingBreeds, setLoadingBreeds] = useState(true);
  const [mapMode, setMapMode] = useState<MapMode>('within_breed');

  const loadingDetail = selectedBreed !== '' && selectedBreed !== loadedBreed;

  // Autocomplete state
  const [query, setQuery] = useState<string>(selectedBreed);
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // Tooltip for map
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [tooltip, setTooltip] = useState<{
    county: string;
    count: number;
    rate: number;
    cancerTypes: { cancer_type: string; count: number }[];
    expanded: boolean;
    x: number;
    y: number;
  } | null>(null);

  const scheduleTooltipClose = () => {
    closeTimerRef.current = setTimeout(() => setTooltip(null), 120);
  };
  const cancelTooltipClose = () => {
    if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
  };

  useEffect(() => {
    fetchFilterOptions()
      .then(opts => {
        const names = [...new Set(opts.breeds.map(b => b.name))].sort();
        setBreeds(names);
      })
      .catch(() => {})
      .finally(() => setLoadingBreeds(false));
  }, []);

  useEffect(() => {
    if (breeds.length > 0 && !breeds.includes(selectedBreed)) {
      onSelectedBreedChange(breeds[0]);
    }
  }, [breeds, onSelectedBreedChange, selectedBreed]);

  useEffect(() => {
    setQuery(selectedBreed);
  }, [selectedBreed]);

  useEffect(() => {
    if (!selectedBreed) return;
    fetchBreedDetail(selectedBreed)
      .then((data) => {
        setDetail(data);
        setLoadedBreed(selectedBreed);
      })
      .catch(() => {
        setDetail(null);
        setLoadedBreed(selectedBreed);
      });
  }, [selectedBreed]);

  // Filtered suggestions
  const suggestions = useMemo(() => {
    if (!query.trim()) return breeds;
    const lower = query.toLowerCase();
    return breeds.filter((b) => b.toLowerCase().includes(lower));
  }, [query, breeds]);

  const effectiveHighlightedIndex = Math.min(highlightedIndex, Math.max(0, suggestions.length - 1));

  useEffect(() => {
    if (isOpen && listRef.current) {
      const item = listRef.current.children[effectiveHighlightedIndex] as HTMLElement | undefined;
      item?.scrollIntoView({ block: 'nearest' });
    }
  }, [effectiveHighlightedIndex, isOpen]);

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
    onSelectedBreedChange(breed);
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
        if (suggestions[effectiveHighlightedIndex]) {
          selectBreed(suggestions[effectiveHighlightedIndex]);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        break;
    }
  };

  const countyCountMap = useMemo(() => {
    const m = new Map<string, number>();
    detail?.county_cases.forEach((c) => m.set(c.county_name.toLowerCase(), c.count));
    return m;
  }, [detail]);

  const countyCancerMap = useMemo(() => {
    const m = new Map<string, { cancer_type: string; count: number }[]>();
    detail?.county_cases.forEach((c) => m.set(c.county_name.toLowerCase(), c.cancer_types));
    return m;
  }, [detail]);

  const countyValueMap = useMemo(() => {
    const m = new Map<string, number>();
    detail?.county_cases.forEach((c) => {
      const denom = mapMode === 'within_breed' ? c.county_breed_tested : c.county_all_tested;
      m.set(c.county_name.toLowerCase(), denom > 0 ? (c.count / denom) * 100 : 0);
    });
    return m;
  }, [detail, mapMode]);

  const colorScale = useMemo(() => {
    return scaleLinear<string>()
      .domain([0, 50, 100])
      .range(['#E6F3F5', '#6BB5BF', '#1A6B77'])
      .clamp(true);
  }, []);

  const maxPccp = detail?.cancer_types[0]?.pccp_within_breed ?? detail?.cancer_types[0]?.count ?? 1;

  if (loadingBreeds) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-sm text-[var(--color-text-secondary)]">Loading breeds…</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* PCCP disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex gap-3">
        <svg className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
        <p className="text-xs text-amber-800 leading-relaxed">
          <span className="font-semibold">PCCP (Pathology-Confirmed Cancer Proportion)</span> — percentage of pathology-tested animals with a confirmed cancer diagnosis.
          Two denominators are shown: <span className="font-medium">% within breed</span> uses only tested animals of that breed; <span className="font-medium">% of all tested</span> uses all tested animals regardless of breed.
          Figures for small cohorts (fewer than 10 tested animals) may be statistically unstable and should be interpreted with caution.
        </p>
      </div>

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
              className="absolute z-40 mt-1 w-full max-h-96 overflow-auto bg-white border border-gray-200 rounded-md shadow-lg"
            >
              {suggestions.map((breed, i) => (
                <li
                  key={breed}
                  onMouseDown={() => selectBreed(breed)}
                  onMouseEnter={() => setHighlightedIndex(i)}
                  className={`px-3 py-2 text-sm cursor-pointer ${
                    i === effectiveHighlightedIndex
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

      {loadingDetail && (
        <div className="flex items-center justify-center h-32">
          <p className="text-sm text-[var(--color-text-secondary)]">Loading breed data…</p>
        </div>
      )}

      {!loadingDetail && detail && (
        <>
          {/* Summary Stats */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
              <span className="font-semibold text-[var(--color-text-primary)]">
                {detail.breed}
              </span>
              {detail.pccp_within_breed != null && detail.breed_total_patients != null ? (
                <>
                  {': '}
                  <span className="font-semibold text-[var(--color-teal-dark)]">
                    {detail.pccp_within_breed.toFixed(1)}%
                  </span>
                  {' within-breed PCCP '}
                  <span className="text-[var(--color-text-secondary)]">
                    ({detail.total_cases} of {detail.breed_total_patients} tested)
                  </span>
                  {detail.pccp_of_all != null && (
                    <span className="text-[var(--color-text-secondary)]">
                      {' · '}
                      <span className="font-medium">{detail.pccp_of_all.toFixed(2)}%</span>
                      {' of all tested (n = '}{detail.global_total_patients?.toLocaleString()}{')'}
                    </span>
                  )}
                </>
              ) : (
                <>
                  {' has '}
                  <span className="font-semibold text-[var(--color-teal-dark)]">
                    {detail.total_cases.toLocaleString()}
                  </span>
                  {' cancer patients'}
                </>
              )}
              {' across '}
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
                  Top cancer types for {detail.breed} · PCCP per 100 tested
                </p>
              </div>
              <div className="p-6">
                {detail.cancer_types.length === 0 ? (
                  <p className="text-sm text-[var(--color-text-secondary)]">
                    No cancer type data available for this breed.
                  </p>
                ) : (
                  <div className="space-y-3">
                    {/* Column headers */}
                    <div className="flex items-center gap-3 mb-1">
                      <span className="w-40 text-[10px] font-medium text-gray-400 uppercase tracking-wider">Cancer Type</span>
                      <div className="flex-1 text-[10px] font-medium text-gray-400 uppercase tracking-wider text-right pr-1">% within breed</div>
                      <span className="w-24 text-[10px] font-medium text-gray-400 uppercase tracking-wider text-right">% of all tested</span>
                    </div>
                    {detail.cancer_types.slice(0, 15).map((ct) => {
                      const primaryVal = ct.pccp_within_breed ?? ct.count;
                      const width = Math.max(5, (primaryVal / maxPccp) * 100);
                      return (
                        <div key={ct.cancer_type} className="flex items-center gap-3">
                          <span className="w-40 text-sm text-[var(--color-text-primary)] truncate" title={ct.cancer_type}>
                            {ct.cancer_type}
                          </span>
                          <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-[var(--color-teal)] to-[var(--color-teal-dark)] rounded-full flex items-center justify-end pr-2"
                              style={{ width: `${width}%` }}
                            >
                              <span className="text-xs font-semibold text-white">
                                {ct.pccp_within_breed != null ? `${ct.pccp_within_breed.toFixed(1)}%` : ct.count.toLocaleString()}
                              </span>
                            </div>
                          </div>
                          <span className="w-24 text-xs text-[var(--color-text-secondary)] text-right tabular-nums">
                            {ct.pccp_of_all != null ? `${ct.pccp_of_all.toFixed(2)}%` : '—'}
                          </span>
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
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                      County Distribution
                    </h3>
                    <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                      Where {detail.breed} cases are located
                    </p>
                  </div>
                  <div className="flex rounded-md border border-gray-200 overflow-hidden text-xs shrink-0">
                    <button
                      onClick={() => setMapMode('within_breed')}
                      className={`px-2 py-1 ${mapMode === 'within_breed' ? 'bg-[var(--color-teal)] text-white font-medium' : 'bg-white text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
                    >
                      PCCP within breed
                    </button>
                    <button
                      onClick={() => setMapMode('of_all')}
                      className={`px-2 py-1 border-l border-gray-200 ${mapMode === 'of_all' ? 'bg-[var(--color-teal)] text-white font-medium' : 'bg-white text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
                    >
                      PCCP of all tested
                    </button>
                  </div>
                </div>
              </div>
              {mapMode === 'within_breed' && (
                <div className="px-3 py-2 bg-blue-50 border-b border-blue-100 flex gap-2">
                  <svg className="w-3.5 h-3.5 text-blue-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  <p className="text-[11px] text-blue-700 leading-relaxed">
                    Rates reflect pathology-tested {detail.breed} animals only. This is not representative of the entire {detail.breed} population in each county.
                    {' '}Note that a patient may have multiple case diagnoses, leading to the sum of cancer patients potentially being lower than the sum of patient cancer types in a county.
                  </p>
                </div>
              )}
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
                        const val = countyValueMap.get(name.toLowerCase()) ?? 0;
                        const fill = val > 0 ? colorScale(val) : '#E5E7EB';

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
                            onClick={() => {
                              setTooltip(prev => prev?.county === name ? { ...prev, expanded: !prev.expanded } : prev);
                            }}
                            onMouseEnter={(e) => {
                              cancelTooltipClose();
                              const event = e as unknown as React.MouseEvent;
                              setTooltip(prev => ({
                                county: name,
                                count: countyCountMap.get(name.toLowerCase()) ?? 0,
                                rate: countyValueMap.get(name.toLowerCase()) ?? 0,
                                cancerTypes: countyCancerMap.get(name.toLowerCase()) ?? [],
                                expanded: prev?.county === name ? prev.expanded : false,
                                x: event.clientX,
                                y: event.clientY,
                              }));
                            }}
                            onMouseLeave={scheduleTooltipClose}
                          />
                        );
                      })
                    }
                  </Geographies>
                </ComposableMap>

                {/* Legend */}
                <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
                  <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">
                    {mapMode === 'within_breed' ? 'PCCP within breed' : 'PCCP of all tested'}
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
                      0%
                    </span>
                    <span className="text-[10px] text-[var(--color-text-secondary)]">
                      100%
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
                    className="fixed z-50"
                    style={{
                      left: tooltip.x + 12,
                      top: tooltip.y - 12,
                      transform: 'translateY(-100%)',
                    }}
                    onMouseEnter={cancelTooltipClose}
                    onMouseLeave={scheduleTooltipClose}
                  >
                    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px] max-w-[220px]">
                      <p className="font-semibold text-sm text-[var(--color-text-primary)]">
                        {tooltip.county}
                      </p>
                      {tooltip.count > 0 ? (
                        <>
                          <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                            {tooltip.count.toLocaleString()} cancer patient{tooltip.count !== 1 ? 's' : ''}
                          </p>
                          <p className="text-xs text-[var(--color-teal-dark)] font-medium mt-0.5">
                            {tooltip.rate.toFixed(2)}%{' '}
                            {mapMode === 'within_breed' ? 'PCCP within breed' : 'PCCP of all tested'}
                          </p>
                          {tooltip.cancerTypes.length > 0 && (
                            <div className="mt-2 pt-2 border-t border-gray-100">
                              <p className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider mb-1">Cancers</p>
                              <ul className="space-y-0.5">
                                {(tooltip.expanded ? tooltip.cancerTypes : tooltip.cancerTypes.slice(0, 3)).map((ct) => (
                                  <li key={ct.cancer_type} className="flex justify-between gap-3 text-xs text-[var(--color-text-primary)]">
                                    <span className="truncate">{ct.cancer_type}</span>
                                    <span className="text-[var(--color-text-secondary)] tabular-nums shrink-0">{ct.count}</span>
                                  </li>
                                ))}
                              </ul>
                              {tooltip.cancerTypes.length > 3 && (
                                <p className="mt-1.5 text-[11px] text-[var(--color-text-secondary)] italic">
                                  {tooltip.expanded
                                    ? 'Click county to collapse'
                                    : `+ ${tooltip.cancerTypes.length - 3} more · click county to expand`}
                                </p>
                              )}
                            </div>
                          )}
                        </>
                      ) : (
                        <p className="text-xs text-[var(--color-text-secondary)] mt-1">No data</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}

      {!loadingDetail && !detail && !selectedBreed && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                Cancer Type Breakdown
              </h3>
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                Top cancer types for selected breed
              </p>
            </div>
            <div className="p-6 flex items-center justify-center h-48">
              <p className="text-sm text-[var(--color-text-secondary)]">Select a breed to view data</p>
            </div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                County Distribution
              </h3>
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                Where selected breed cases are located
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
                    geographies.map((geo) => (
                      <Geography
                        key={geo.rsmKey}
                        geography={geo}
                        fill="#E5E7EB"
                        stroke="#FFFFFF"
                        strokeWidth={0.5}
                        style={{
                          default: { outline: 'none' },
                          hover: { outline: 'none' },
                          pressed: { outline: 'none' },
                        }}
                      />
                    ))
                  }
                </Geographies>
              </ComposableMap>
              <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
                  <span className="text-[10px] text-[var(--color-text-secondary)]">No data</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {!loadingDetail && !detail && selectedBreed && (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
          <p className="text-sm text-[var(--color-text-secondary)]">
            No data available for {selectedBreed}.
          </p>
        </div>
      )}
    </div>
  );
}
